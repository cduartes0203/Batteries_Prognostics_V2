import numpy as np
from Functions.Utils_RTLO import *

class RTLO:
    def __init__(self, nI,nR,nO,ηS=[0.1,0.1,0.1], τ=10,mode='past'):
        np.random.seed(42)
        self.k = 1
        self.j = nI-1
        self.t = np.array([])
        self.start = 0
        self.ref = None
        self.act = 'tanh'
        self.flw = mode
        self.n = -1
        self.nI, self.nR, self.nO = nI, nR, nO

        self.ηS = np.array(ηS)
        self.τ = τ
        self.ρ = 0.003

        self.x = np.zeros(nI)
        self.hP,self.hP, self.hU, self.hL = [0.1*np.ones(nR) for i in range(4)]
        self.hP2, self.hU2, self.hL2 = [0.1*np.ones(nR) for i in range(3)]
        self.uP, self.uP2 =  [np.zeros(nR) for i in range(2)]
        self.pS = np.zeros((self.nR, self.nR))
        self.qS = np.zeros((self.nR, self.nI))
        
        self.wI = XavierUniform([nR, nI],sd=42)
        self.wR = XavierUniform([nR, nR],sd=41)
        self.wO = XavierUniform([nO, nR],sd=40)
        self.BS = XavierUniform([nR, nO],sd=39)
    
        self.yP, self.yR, self.yL, self.yU = [np.array([]) for i in range(4)]
        self.yP_hist = np.zeros(self.nI)
        self.εY, self.εM, self.εR, self.εE, self.eP, self.eR, self.ΣW = [0 for i in range(7)]
        self.εM_hist,self.εR_hist, self.eR_hist, self.eP_hist = [np.array([]) for i in range(4)]
        self.wR_hist = []
        self.wI_hist = []
        self.wO_hist = []

        self.rR = 1e-9
        self.rP = 1e-10
        self.rL = 1e-11
        self.rU = 1e-12
        self.rRsum = 0
        self.rulR, self.rulP, self.rulL, self.rulU = [np.array([]) for i in range(4)]


    def PredSingle(self,x):

        u = np.dot(self.wR, self.hP) + np.dot(self.wI, x)
        h = self.hP + (-self.hP + Activation(u,self.act))/self.τ
        y = np.dot(self.wO, h)

        return y

    def fit(self,xP,yR):
        if self.flw != 'past': self.n = 0
        
        η1,η2,η3 = self.ηS      

        uP = self.wR @ self.hP + self.wI @ xP
        hP = self.hP*(1-1/self.τ) +Activation(uP,self.act)/self.τ
        yP = self.wO @ hP
        eS = yR-yP

        self.pS = np.outer(dActivation(uP,self.act),self.hP)/self.τ + (1-1/self.τ)*self.pS
        self.qS = np.outer(dActivation(uP,self.act),self.x)/self.τ + (1-1/self.τ)*self.qS

        δOS = η1*np.outer(eS,hP)
        δRS = η2*np.outer((self.BS@eS),np.ones(self.nR))*self.pS
        δIS = η3*np.outer(np.dot(self.BS, eS),np.ones(self.nI))*self.qS

        self.wI = self.wI + δIS
        self.wR = self.wR + δRS
        self.wO = self.wO + δOS

        self.wR_hist.append(self.wR.flatten())
        self.wI_hist.append(self.wI.flatten())
        self.wO_hist.append(self.wO.flatten())

        self.hP = hP
        self.uP = uP
        self.x = xP

        self.k = self.k+1
        #self.ηS = self.ηS/(1 + self.decay*self.k

    def PredRul(self, x,maxRul=100,lim=0.2,store=False):
        if self.flw != 'past': self.n = 0
        xP = x.copy()
        rulP=0
        predict = True
        hP = self.hP2.copy()
        while predict:
            uP = (self.wR @ hP) + (self.wI @ xP)
            hP = hP*(1-1/self.τ) + Activation(uP,self.act)/self.τ
            yP = (self.wO @ hP)[self.n]
            xP = np.delete(np.append(xP,yP),0)

            if predict: rulP = rulP+1
            if yP < lim: predict = False
            if rulP >= maxRul:
                rulP=1
                break
        self.rR=self.ref-self.k
        self.rP = rulP

        if store:
            if self.k>=self.start:
                self.rulR = np.append(self.rulR,self.rR)
                self.rulP = np.append(self.rulP,self.rP)

    def PredRulIntr(self, x,lim=0.2,maxRul=110,store=False,show=False):
        if self.flw != 'past': self.n = 0
        xP,xL,xU =x.copy(), x.copy(), x.copy()
        k = 1
        predict = True
        PredRuls = [True for i in range(3)]
        PredVals, Ruls = [0 for i in range(3)], [0 for i in range(3)]
        wR,wI,wO = self.wR,self.wI,self.wO
        hP,hU,hL = [self.hP.copy() for i in range(3)]

        wRU, wRL = np.maximum((1+self.ρ)*wR, wR/(1+self.ρ)), np.minimum((1+self.ρ)*wR, wR/(1+self.ρ))
        wIU, wIL = np.maximum((1+self.ρ)*wI, wI/(1+self.ρ)), np.minimum((1+self.ρ)*wI, wI/(1+self.ρ))
        wOU, wOL = np.maximum((1+self.ρ)*wO, wO/(1+self.ρ)), np.minimum((1+self.ρ)*wO, wO/(1+self.ρ)) 
  
        while predict:

            uP = ( wR @ hP) + ( wI @ xP)
            uL = (wRL @ hL) + (wIL @ xL)
            uU = (wRU @ hU) + (wIU @ xU)
            uU, uL = np.maximum(uU,uL), np.minimum(uU,uL)
            
            hP = hP*(1-1/self.τ) + Activation(uP,self.act)/self.τ
            hL = hL*(1-1/self.τ) + Activation(uL,self.act)/self.τ
            hU = hU*(1-1/self.τ) + Activation(uU,self.act)/self.τ
            hU, hL = np.maximum(hU,hL), np.minimum(hU,hL)

            yP = ( wO @ hP)
            yL = (wOL @ hL)
            yU = (wOU @ hU)
            yU, yL = np.maximum(yU, yL), np.minimum(yU, yL)

            if show:   
                print(k,yL,yP,yU)

            PredVals = ([yL[self.n],yP[self.n],yU[self.n]])
            yL,yP,yU = PredVals

            xP = np.delete(np.append(xP,yP),0)
            xL = np.delete(np.append(xL,yL),0)
            xU = np.delete(np.append(xU,yU),0)

            if Ruls[0] == 0:
                if self.k>=self.start:
                    self.yL = np.append(self.yL,yL)
                    self.yP = np.append(self.yP,yP)
                    self.yU = np.append(self.yU,yU)
            
            CheckPred,CheckLim=0,0
            for i in range(3):
                if PredRuls[i]: 
                    Ruls[i] = Ruls[i]+1
                    if Ruls[i] >= maxRul:
                        CheckLim = CheckLim + 1
                        Ruls[i] = 1
                        PredRuls[i] = False
                if PredVals[i] < lim: PredRuls[i] = False
                if not PredRuls[i]: CheckPred = CheckPred + 1
            if CheckPred == 3:break
            if CheckLim == 3:break
            k = k+1

        if Ruls[1]< Ruls[0]: Ruls[1]=Ruls[0]   
        if Ruls[2]< Ruls[1]: Ruls[2]=Ruls[1]  +1      
        self.rR=self.ref-self.k
        self.rL,self.rP,self.rU = Ruls

        if store:
            if self.k>=self.start:
                self.rulR = np.append(self.rulR,self.rR)
                self.rulL = np.append(self.rulL,self.rL)
                self.rulP = np.append(self.rulP,self.rP)
                self.rulU = np.append(self.rulU,self.rU)
    
    def Predict(self, x):

        if self.flw != 'past': self.n = 0
        xP = x.copy()
        
        hP = self.hP2.copy()

        uP = (self.wR @ hP) + (self.wI @ xP)
        hP = hP*(1-1/self.τ) + Activation(uP,self.act)/self.τ
        yP = (self.wO @ hP)[self.n]

        self.hP2 = hP

        return yP

    def Restore(self):

        self.hP2 = self.hP
        self.hL2 = self.hP
        self.hU2 = self.hP

    def ReturnParameters(self):

        return [self.wR,self.wI,self.wO,self.pS,self.qS,self.hP,self.hP2,self.x]
    
    def ReceiveParameters(self,vec):
        self.wR,self.wI,self.wO,self.pS,self.qS,self.hP,self.hP2,self.x = vec
        



 
        
