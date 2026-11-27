import copy
from collections import deque
import numpy as np
from Testing import eRNN_R4 as _R4
from Testing.eRNN_R4 import *     # XavierUniform, WeightInterval, IntervalMatVec, Activation, dActivation


class eRNN(_R4.eRNN):
    '''
    eRNN_R4 + treino em malha fechada (closed-loop) por repetição causal (replay) de uma previsão livre.

    Motivação: o treino do R4 (fit) é sempre com entradas observadas (teacher forcing), mas a RUL é obtida por previsão
    recursiva longa, em que a RNN recebe as próprias previsões como entrada. Nada no treino corrige essa previsão livre.

    Treino em malha fechada (ClosedLoopFit), chamado após fit no mesmo passo k:
        1) fit(xP, yR, t) guarda no buffer (até cl_H entradas): instante t, janela observada xP, estado recorrente hP
           ANTES do passo e o alvo um passo à frente yR[n] (= h_{t+1}, já conhecido quando fit é chamado).
        2) Usa o trecho final contíguo do buffer (instantes consecutivos t_0, ..., t_{m-1}, m <= cl_H, m >= 2):
           parte da janela observada x_{t_0} e do estado hP guardado em t_0 e roda m passos livres; a partir do 2º passo
           a entrada é construída com a previsão anterior (next_input(x, y_n), fornecida pelo AutoCloud: depende da
           representação do HI, nível ou incrementos).
        3) Em cada passo livre i: erro só na coordenada realimentada n (alvo h_{t_i+1}), traços de elegibilidade locais
           (começam em zero, mesma regra do fit: RFLO) e sinais de aprendizado (LearningSignals); as variações dos pesos
           são acumuladas e aplicadas no fim, em média sobre os m passos e com as taxas η multiplicadas por cl_eta.
        4) O estado recorrente real (hP), os traços do fit e o buffer não são alterados pela repetição.
    Causal: todos os alvos usados (h_{t_i+1}, t_i <= k) são conhecidos no passo k; nenhum valor futuro é usado.
    Aproximação: a previsão realimentada entra como constante (sem gradiente através da realimentação), como o RFLO
    já trunca o gradiente no tempo.
    cl_H = 0 (padrão): sem buffer nem repetição; resultados idênticos ao eRNN_R4.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SetClosedLoop(0, 1.0)

    def SetClosedLoop(self, H, eta=1.0):
        '''Configura o treino em malha fechada: H = nº máximo de passos da repetição (0 = desligado);
        eta = multiplicador das taxas η nas atualizações da repetição.'''
        H = int(H)
        if H < 0: raise ValueError('cl_H deve ser >= 0.')
        if not eta >= 0: raise ValueError('cl_eta deve ser >= 0.')
        self.cl_H, self.cl_eta = H, float(eta)
        self.cl_buf = deque(maxlen=H) if H > 0 else None
        self.cl_err = np.nan          # erro absoluto médio da última repetição (diagnóstico)

    def ClearReplay(self):
        '''Esvazia o buffer da repetição (novo rolamento: as janelas do anterior não são contíguas às do novo).'''
        if self.cl_buf is not None: self.cl_buf.clear()

    def fit(self, xP, yR, t=None):
        if self.cl_H > 0 and t is not None:
            self.cl_buf.append((int(t), np.array(xP, dtype=float).copy(), [h.copy() for h in self.hP],
                                float(yR[self.n])))
        super().fit(xP, yR)

    def ClosedLoopFit(self, next_input):
        '''Repetição em malha fechada sobre o trecho final contíguo do buffer (ver docstring da classe).'''
        if self.cl_H <= 0 or self.cl_buf is None or len(self.cl_buf) < 2: return
        buf = list(self.cl_buf)
        i0 = len(buf) - 1
        while i0 > 0 and buf[i0 - 1][0] == buf[i0][0] - 1: i0 -= 1
        seg = buf[i0:]
        m = len(seg)
        if m < 2: return

        η1, η2, η3 = self.ηS * self.cl_eta
        a = 1 - 1 / self.τ
        x = seg[0][1].copy()
        hP = [h.copy() for h in seg[0][2]]
        pS = [np.zeros_like(p) for p in self.pS]
        qS = [np.zeros_like(q) for q in self.qS]
        dWR = [np.zeros_like(w) for w in self.wR]
        dWI = [np.zeros_like(w) for w in self.wI]
        dWO = np.zeros_like(self.wO)
        errs = []
        for i, (_, _, _, target) in enumerate(seg):
            curr_in = np.append(x, 1)
            new_in, new_uP, new_hP = [], [], []
            for l in range(self.nL):
                u = self.wR[l] @ hP[l] + self.wI[l] @ curr_in
                h = hP[l] * a + Activation(u, self.act) / self.τ
                new_in.append(curr_in)
                new_uP.append(u)
                new_hP.append(h)
                curr_in = np.append(h, 1)
            y = self.wO @ new_hP[-1]
            eS = np.zeros(self.nO)
            eS[self.n] = target - y[self.n]          # erro apenas na coordenada realimentada (h_{t+1})
            errs.append(abs(eS[self.n]))
            dφ = [dActivation(u, self.act) for u in new_uP]
            for l in range(self.nL):
                pS[l] = np.outer(dφ[l], hP[l]) / self.τ + a * pS[l]
                qS[l] = np.outer(dφ[l], new_in[l]) / self.τ + a * qS[l]
            lS = self.LearningSignals(eS, dφ)
            for l in range(self.nL):
                dWR[l] += lS[l][:, None] * pS[l]
                dWI[l] += lS[l][:, None] * qS[l]
            dWO += np.outer(eS, new_hP[-1])
            hP = new_hP
            x = next_input(x, y[self.n])             # próxima entrada com a própria previsão (malha fechada)

        for l in range(self.nL):
            self.wR[l] += η2 * dWR[l] / m
            self.wI[l] += η3 * dWI[l] / m
        self.wO += η1 * dWO / m
        self.cl_err = float(np.mean(errs))
