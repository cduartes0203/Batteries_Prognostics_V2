import numpy as np
from Functions.Utils_RTLO import *

class RTLO:
    def __init__(self, nI, nR, nO, ηS=[0.1, 0.1, 0.1], τ=10, mode='past', act='tanh'):
        np.random.seed(42)
        
        if isinstance(nR, (int, np.integer)):
            self.nR = [nR]
        else:
            self.nR = list(nR)
            
        self.n_layers = len(self.nR)
        
        self.k = 1
        self.j = nI - 1
        self.t = np.array([])
        self.start = 0
        self.ref = None
        self.act = act
        self.flw = mode
        self.n = -1
        self.nI, self.nO = nI + 1, nO

        self.ηS = np.array(ηS)
        self.τ = τ
        self.ρ = 0.003

        self.x = np.zeros(nI + 1)
        
        self.hP = [0.1 * np.ones(size) for size in self.nR]
        self.hU = [0.1 * np.ones(size) for size in self.nR]
        self.hL = [0.1 * np.ones(size) for size in self.nR]
        
        self.hP2 = [0.1 * np.ones(size) for size in self.nR]
        self.hU2 = [0.1 * np.ones(size) for size in self.nR]
        self.hL2 = [0.1 * np.ones(size) for size in self.nR]
        
        self.uP = [np.zeros(size) for size in self.nR]
        self.uP2 = [np.zeros(size) for size in self.nR]
        
        self.pS = [np.zeros((self.nR[l], self.nR[l])) for l in range(self.n_layers)]
        self.qS = [np.zeros((self.nR[l], (nI + 1) if l == 0 else (self.nR[l-1] + 1))) for l in range(self.n_layers)]

        # Pesos de Entrada e Inter-Camadas
        self.wI = []
        # Camada 0: recebe a entrada externa (nI + 1)
        self.wI.append(XavierUniform([self.nR[0], nI + 1], sd=42))
        # Camadas seguintes l: recebem a saída da camada l-1 mais o termo de bias (+1)
        for l in range(1, self.n_layers):
            self.wI.append(XavierUniform([self.nR[l], self.nR[l-1] + 1], sd=42 + l))

        # Pesos Recorrentes para cada camada
        self.wR = [XavierUniform([self.nR[l], self.nR[l]], sd=41 + l) for l in range(self.n_layers)]
        
        # Pesos de Saída conectados à ÚLTIMA camada oculta
        self.wO = XavierUniform([nO, self.nR[-1]], sd=40)
        
        # Feedback de Erro
        self.BS = [XavierUniform([self.nR[l], nO], sd=39 + l) for l in range(self.n_layers)]

        self.yP, self.yR, self.yL, self.yU = [np.array([]) for i in range(4)]
        self.yP_hist = np.zeros(self.nI)
        self.εY, self.εM, self.εR, self.εE, self.eP, self.eR, self.ΣW = [0 for i in range(7)]
        self.εM_hist, self.εR_hist, self.eR_hist, self.eP_hist = [np.array([]) for i in range(4)]
        
        self.wR_hist = [[] for _ in range(self.n_layers)]
        self.wI_hist = [[] for _ in range(self.n_layers)]
        self.wO_hist = []

        self.rR = 1e-9
        self.rP = 1e-10
        self.rL = 1e-11
        self.rU = 1e-12
        self.rRsum = 0
        self.rulR, self.rulP, self.rulL, self.rulU = [np.array([]) for i in range(4)]

    def PredSingle(self, x):
        curr_in = np.append(x, 1)
        h_temp = [h.copy() for h in self.hP]
        
        for l in range(self.n_layers):
            u = np.dot(self.wR[l], h_temp[l]) + np.dot(self.wI[l], curr_in)
            h = h_temp[l] + (-h_temp[l] + Activation(u, self.act)) / self.τ
            h_temp[l] = h
            curr_in = np.append(h, 1)
            
        y = np.dot(self.wO, h_temp[-1])
        return y

    def fit(self, xP, yR):
        if self.flw != 'past': self.n = 0
        xP = np.append(xP, 1)
        η1, η2, η3 = self.ηS
        
        curr_in = xP
        new_uP = []
        new_hP = []

        # Passagem Direta (Forward) através das camadas ocultas
        for l in range(self.n_layers):
            u = self.wR[l] @ self.hP[l] + self.wI[l] @ curr_in
            h = self.hP[l] * (1 - 1 / self.τ) + Activation(u, self.act) / self.τ
            new_uP.append(u)
            new_hP.append(h)
            curr_in = np.append(h, 1)

        # Cálculo da Saída usando a última camada
        yP = self.wO @ new_hP[-1]
        eS = yR - yP

        # Atualização dos Traços e Pesos
        curr_x = xP
        for l in range(self.n_layers):
            self.pS[l] = np.outer(dActivation(self.uP[l], self.act), self.hP[l]) / self.τ + (1 - 1 / self.τ) * self.pS[l]
            self.qS[l] = np.outer(dActivation(self.uP[l], self.act), curr_x) / self.τ + (1 - 1 / self.τ) * self.qS[l]

            δRS = η2 * np.outer((self.BS[l] @ eS), np.ones(self.nR[l])) * self.pS[l]
            δIS = η3 * np.outer(np.dot(self.BS[l], eS), np.ones(self.qS[l].shape[1])) * self.qS[l]

            self.wI[l] += δIS
            self.wR[l] += δRS

            self.wR_hist[l].append(self.wR[l].flatten())
            self.wI_hist[l].append(self.wI[l].flatten())
            
            curr_x = np.append(new_hP[l], 1)

        # Pesos da Saída
        δOS = η1 * np.outer(eS, new_hP[-1])
        self.wO += δOS
        self.wO_hist.append(self.wO.flatten())

        # Atualização de Estados
        self.hP = [h.copy() for h in new_hP]
        self.hP2 = [h.copy() for h in new_hP]
        self.hL = [h.copy() for h in new_hP]
        self.hU = [h.copy() for h in new_hP]
        self.uP = new_uP
        self.x = xP

        self.k += 1

    def Predict(self, x):
        if self.flw != 'past': self.n = 0
        xP = np.append(x.copy(), 1)
        
        curr_in = xP
        for l in range(self.n_layers):
            hP = self.hP2[l].copy()
            uP = (self.wR[l] @ hP) + (self.wI[l] @ curr_in)
            hP = hP * (1 - 1 / self.τ) + Activation(uP, self.act) / self.τ
            self.hP2[l] = hP
            curr_in = np.append(hP, 1)

        yP = (self.wO @ self.hP2[-1])[self.n]
        return yP

    def PredictionVector(self, x):
        if self.flw != 'past': self.n = 0
        xP = np.append(x.copy(), 1)
        
        curr_in = xP
        for l in range(self.n_layers):
            hP = self.hP2[l].copy()
            uP = (self.wR[l] @ hP) + (self.wI[l] @ curr_in)
            hP = hP * (1 - 1 / self.τ) + Activation(uP, self.act) / self.τ
            self.hP2[l] = hP
            curr_in = np.append(hP, 1)

        yP = (self.wO @ self.hP2[-1])
        return yP
    
    def PredictSingle(self, xP, show=False):
        xP = np.append(xP, 1)
        if self.flw != 'past': self.n = 0
        
        curr_in = xP
        for l in range(self.n_layers):
            hP = self.hP2[l].copy()
            uP = (self.wR[l] @ hP) + (self.wI[l] @ curr_in)
            hP = hP * (1 - 1 / self.τ) + Activation(uP, self.act) / self.τ
            self.hP2[l] = hP
            curr_in = np.append(hP, 1)

        yP = (self.wO @ self.hP2[-1])
        yP = yP[self.n]
        return yP

    def PredictIntr(self, xP, xL, xU, ep, show=False):
        xP = np.append(xP, 1)
        xL = np.append(xL, 1)
        xU = np.append(xU, 1)
        if self.flw != 'past': self.n = 0

        curr_inP = xP
        curr_inL = xL
        curr_inU = xU

        wO = self.wO
        wOU, wOL = np.maximum((1 + ep) * wO, wO / (1 + ep)), np.minimum((1 + ep) * wO, wO / (1 + ep))

        for l in range(self.n_layers):
            wR, wI = self.wR[l], self.wI[l]
            hP, hL, hU = self.hP2[l].copy(), self.hL2[l].copy(), self.hU2[l].copy()

            wRU, wRL = np.maximum((1 + ep) * wR, wR / (1 + ep)), np.minimum((1 + ep) * wR, wR / (1 + ep))
            wIU, wIL = np.maximum((1 + ep) * wI, wI / (1 + ep)), np.minimum((1 + ep) * wI, wI / (1 + ep))

            uP = (wR @ hP) + (wI @ curr_inP)
            uL = (wRL @ hL) + (wIL @ curr_inL)
            uU = (wRU @ hU) + (wIU @ curr_inU)

            uU, uL = np.maximum(uU, uL), np.minimum(uU, uL)

            hP = hP * (1 - 1 / self.τ) + Activation(uP, self.act) / self.τ
            hL = hL * (1 - 1 / self.τ) + Activation(uL, self.act) / self.τ
            hU = hU * (1 - 1 / self.τ) + Activation(uU, self.act) / self.τ

            hU, hL = np.maximum(hU, hL), np.minimum(hU, hL)

            self.hP2[l] = hP
            self.hL2[l] = hL
            self.hU2[l] = hU

            curr_inP = np.append(hP, 1)
            curr_inL = np.append(hL, 1)
            curr_inU = np.append(hU, 1)

        yP = (wO @ self.hP2[-1])
        yL = (wOL @ self.hL2[-1])
        yU = (wOU @ self.hU2[-1])

        yU, yL = np.maximum(yU, yL), np.minimum(yU, yL)

        yP = yP[self.n]
        yL = yL[self.n]
        yU = yU[self.n]

        if show:
            print('-----------------')

        return np.array([yL, yP, yU])

    def Restore(self):
        self.hP2 = [h.copy() for h in self.hP]
        self.hL2 = [h.copy() for h in self.hP]
        self.hU2 = [h.copy() for h in self.hP]

    def ReturnParameters(self):
        return [self.wR, self.wI, self.wO, self.pS, self.qS, self.hP, self.hP2, self.hL2, self.hU2, self.x]
        #return self.wR, self.wI, self.wO, self.pS, self.qS, self.hP, self.hP2, self.hL2, self.hU2, self.x

    def ReceiveParameters(self, vec):
        self.wR, self.wI, self.wO, self.pS, self.qS, self.hP, self.hP2, self.hL2, self.hU2, self.x = vec