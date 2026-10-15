import numpy as np
from Functions.Utils_RTLO import *

class RTLO:
    def __init__(self, nI, nR, nO, ηS=[0.1], τ=10, mode='past', act='tanh', m_history=10):
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

        # Pesos de Entrada e Inter-Camadas
        self.wI = []
        self.wI.append(XavierUniform([self.nR[0], nI + 1], sd=42))
        for l in range(1, self.n_layers):
            self.wI.append(XavierUniform([self.nR[l], self.nR[l-1] + 1], sd=42 + l))

        # Pesos Recorrentes
        self.wR = [XavierUniform([self.nR[l], self.nR[l]], sd=41 + l) for l in range(self.n_layers)]
        
        # Pesos de Saída
        self.wO = XavierUniform([nO, self.nR[-1]], sd=40)

        self.yP, self.yR, self.yL, self.yU = [np.array([]) for i in range(4)]
        self.yP_hist = np.zeros(self.nI)
        self.εY, self.εM, self.εR, self.εE, self.eP, self.eR, self.ΣW = [0 for i in range(7)]
        self.εM_hist, self.εR_hist, self.eR_hist, self.eP_hist = [np.array([]) for i in range(4)]
        
        self.wR_hist = [[] for _ in range(self.n_layers)]
        self.wI_hist = [[] for _ in range(self.n_layers)]
        self.wO_hist = []

        # =====================================================================
        # ESTRUTURAS DE MEMÓRIA DO L-BFGS
        # =====================================================================
        self.m_history = m_history
        self.s_history = []  # s_k = w_k - w_{k-1}
        self.y_history = []  # y_k = g_k - g_{k-1}
        self.g_prev = None
        self.w_prev = None

    def _pack_params(self):
        """ Achata e empacota todos os parâmetros wO, wI e wR em um vetor único """
        params = [self.wO.flatten()]
        for l in range(self.n_layers):
            params.append(self.wI[l].flatten())
            params.append(self.wR[l].flatten())
        return np.concatenate(params)

    def _unpack_params(self, w_vec):
        """ Desempacota o vetor único de volta para os atributos de pesos do modelo """
        idx = 0
        shape_wO, size_wO = self.wO.shape, self.wO.size
        self.wO = w_vec[idx:idx+size_wO].reshape(shape_wO)
        idx += size_wO

        for l in range(self.n_layers):
            shape_wI, size_wI = self.wI[l].shape, self.wI[l].size
            self.wI[l] = w_vec[idx:idx+size_wI].reshape(shape_wI)
            idx += size_wI

            shape_wR, size_wR = self.wR[l].shape, self.wR[l].size
            self.wR[l] = w_vec[idx:idx+size_wR].reshape(shape_wR)
            idx += size_wR

    def _lbfgs_two_loop(self, g_k):
        """ Algoritmo L-BFGS Two-Loop Recursion """
        q = g_k.copy()
        alphas = []
        k_mem = len(self.s_history)

        # Loop 1: Direção Regressiva
        for i in reversed(range(k_mem)):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
            
            alpha_i = rho_i * np.dot(s_i, q)
            alphas.append(alpha_i)
            q -= alpha_i * y_i

        alphas.reverse()

        # Factor de Escalonamento Inicial gamma_k
        if k_mem > 0:
            s_last = self.s_history[-1]
            y_last = self.y_history[-1]
            gamma_k = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-10)
        else:
            gamma_k = 1.0

        r = gamma_k * q

        # Loop 2: Direção Progressiva
        for i in range(k_mem):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
            
            beta = rho_i * np.dot(y_i, r)
            r += s_i * (alphas[i] - beta)

        return -r  # Direção de atualização d_k

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
        
        curr_in = xP
        new_uP = []
        new_hP = []
        
        # Entradas acumuladas em cada camada para o cálculo do gradiente
        layer_inputs = [curr_in]

        # 1. Forward Pass
        for l in range(self.n_layers):
            u = self.wR[l] @ self.hP[l] + self.wI[l] @ curr_in
            h = self.hP[l] * (1 - 1 / self.τ) + Activation(u, self.act) / self.τ
            new_uP.append(u)
            new_hP.append(h)
            curr_in = np.append(h, 1)
            layer_inputs.append(curr_in)

        yP = self.wO @ new_hP[-1]
        eS = yR - yP  # Erro de saída (yR - yP)

        # 2. Cálculo Exato dos Gradientes (Loss = 0.5 * ||eS||^2 => dL/dyP = -eS)
        g_O = - np.outer(eS, new_hP[-1])

        grad_wI = [None] * self.n_layers
        grad_wR = [None] * self.n_layers

        # Propagação do gradiente a partir da última camada oculta
        dh_next = - (self.wO.T @ eS)

        for l in reversed(range(self.n_layers)):
            # Derivada da ativação local
            du = (dh_next / self.τ) * dActivation(new_uP[l], self.act)
            
            # Gradientes exatos em relação a wR e wI
            grad_wR[l] = np.outer(du, self.hP[l])
            grad_wI[l] = np.outer(du, layer_inputs[l])

            # Propagação do sinal de erro para a camada anterior (se houver)
            if l > 0:
                # Retira a componente do bias ao retropropagar
                dh_next = self.wI[l][:, :-1].T @ du

        # Concatenar todos os gradientes em um único vetor g_k
        grad_list = [g_O.flatten()]
        for l in range(self.n_layers):
            grad_list.append(grad_wI[l].flatten())
            grad_list.append(grad_wR[l].flatten())
        
        g_k = np.concatenate(grad_list)
        w_k = self._pack_params()

        # 3. Atualizar Histórico de Memória do L-BFGS (s_k e y_k)
        if self.g_prev is not None and self.w_prev is not None:
            s_k = w_k - self.w_prev
            y_k = g_k - self.g_prev

            if np.dot(y_k, s_k) > 1e-8:  # Validação de curvatura
                if len(self.s_history) >= self.m_history:
                    self.s_history.pop(0)
                    self.y_history.pop(0)
                self.s_history.append(s_k)
                self.y_history.append(y_k)

        # 4. Direção L-BFGS Purcell
        d_k = self._lbfgs_two_loop(g_k)

        # 5. Atualização dos Pesos
        η = self.ηS[0]
        w_new = w_k + η * d_k

        self.w_prev = w_k.copy()
        self.g_prev = g_k.copy()

        self._unpack_params(w_new)

        # Salvar histórico de pesos
        for l in range(self.n_layers):
            self.wR_hist[l].append(self.wR[l].flatten())
            self.wI_hist[l].append(self.wI[l].flatten())
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
        return [self.wR, self.wI, self.wO, self.hP, self.hP2, self.hL2, self.hU2, self.x]

    def ReceiveParameters(self, vec):
        self.wR, self.wI, self.wO, self.hP, self.hP2, self.hL2, self.hU2, self.x = vec