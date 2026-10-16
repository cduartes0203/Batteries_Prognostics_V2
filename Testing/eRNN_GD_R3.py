import copy
import numpy as np
from Functions.Utils_RTLO import Activation, dActivation


def XavierUniform(shape, sd):
    # Gerador local: não altera o estado global do np.random
    rng = np.random.default_rng(sd)
    n_in, n_out = shape
    limit = np.sqrt(6 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=shape)


def WeightInterval(W, ep):
    # Grânulo do peso (protocolo 2 de Al-Hmouz et al., 2024): [min(w/(1+ε), w(1+ε)), max(w/(1+ε), w(1+ε))]
    a, b = (1 + ep) * W, W / (1 + ep)
    return np.minimum(a, b), np.maximum(a, b)


def IntervalMatVec(WL, WU, xL, xU):
    # Produto intervalar exato [WL, WU] @ [xL, xU]: para cada w_ij * x_j, mínimo e máximo dos
    # 4 produtos dos extremos; a soma preserva a inclusão (contém o produto de quaisquer pontos internos)
    c = np.stack([WL * xL, WL * xU, WU * xL, WU * xU])
    return c.min(axis=0).sum(axis=1), c.max(axis=0).sum(axis=1)


class eRNN:
    '''
    Deep leaky RNN trained online with RFLO eligibility traces (Murray, 2019).

    Layer l at step t (h^0(t) = x(t), [v;1] appends the bias input):
        u^l(t) = wR[l] h^l(t-1) + wI[l] [h^{l-1}(t); 1]
        h^l(t) = (1 - 1/τ) h^l(t-1) + φ(u^l(t))/τ
        y(t)   = wO h^L(t)

    feedback:
        'symmetric': learning signal of each layer is the true error
                     gradient within the current step
                     (ℓ^L = wO^T e, ℓ^l = wI[l+1][:, :-1]^T (φ'(u^{l+1}) ⊙ ℓ^{l+1}) / τ).
        'random'   : fixed random matrices ℓ^l = B^l e (RFLO + direct feedback alignment).

    ηS: [η_out, η_rec, η_in]; values >= 1 are read as exponents N -> 10^-N.

    nPast: number of past samples at the start of the output window in 'ahead' mode
           (mO in PrepareData). The one-step-ahead prediction is output[nPast].
    '''

    def __init__(self, nI, nR, nO, ηS=None, τ=10, sd=42, mode='past', act='tanh',
                 feedback='symmetric', store_hist=False, nPast=0):

        if isinstance(nR, (int, np.integer)):
            self.nR = [int(nR)]
        else:
            self.nR = [int(n) for n in nR]
        self.nL = len(self.nR)

        if mode == 0: mode = 'past'
        elif mode == 1: mode = 'ahead'
        if act == 0: act = 'tanh'
        elif act == 1: act = 'relu'
        elif act == 2: act = 'sigmoid'
        if feedback not in ('symmetric', 'random'):
            raise ValueError("feedback deve ser 'symmetric' ou 'random'.")

        self.flw = mode
        self.act = act
        self.feedback = feedback
        self.store_hist = store_hist
        # Índice da predição um passo à frente: último em 'past'; em 'ahead' a janela de saída
        # começa com nPast valores passados, logo o próximo valor está no índice nPast
        if not 0 <= int(nPast) < nO:
            raise ValueError('nPast deve satisfazer 0 <= nPast < nO.')
        self.n = -1 if mode == 'past' else int(nPast)
        self.k = 1
        self.nI, self.nO = nI + 1, nO

        # Cópia em float: não altera o objeto passado pelo chamador
        if ηS is None: ηS = [0.1, 0.1, 0.1]
        ηS = np.array(ηS, dtype=float).ravel()
        if ηS.size != 3:
            raise ValueError('ηS deve conter 3 taxas: [η_out, η_rec, η_in].')
        self.ηS = np.where(ηS >= 1, 10.0 ** (-ηS), ηS)

        self.τ = τ
        self.x = np.zeros(nI + 1)

        self.hP = [0.1 * np.ones(size) for size in self.nR]
        self.hP2 = [0.1 * np.ones(size) for size in self.nR]
        self.hL2 = [0.1 * np.ones(size) for size in self.nR]
        self.hU2 = [0.1 * np.ones(size) for size in self.nR]
        self.uP = [np.zeros(size) for size in self.nR]

        self.pS = [np.zeros((self.nR[l], self.nR[l])) for l in range(self.nL)]
        self.qS = [np.zeros((self.nR[l], (nI + 1) if l == 0 else (self.nR[l-1] + 1))) for l in range(self.nL)]

        self.wI = [XavierUniform([self.nR[0], nI + 1], sd=sd)]
        for l in range(1, self.nL):
            self.wI.append(XavierUniform([self.nR[l], self.nR[l-1] + 1], sd=sd + l))
        self.wR = [XavierUniform([self.nR[l], self.nR[l]], sd=sd - 1 + l) for l in range(self.nL)]
        self.wO = XavierUniform([nO, self.nR[-1]], sd=sd - 2)

        if feedback == 'random':
            self.BS = [XavierUniform([self.nR[l], nO], sd=sd - 3 + l) for l in range(self.nL)]
        else:
            self.BS = None

        self.wR_hist = [[] for _ in range(self.nL)]
        self.wI_hist = [[] for _ in range(self.nL)]
        self.wO_hist = []

    def LearningSignals(self, eS, dφ):
        if self.feedback == 'random':
            return [B @ eS for B in self.BS]

        # Gradiente exato dentro do passo atual, propagado da saída às camadas inferiores
        lS = [None] * self.nL
        lS[-1] = self.wO.T @ eS
        for l in range(self.nL - 2, -1, -1):
            lS[l] = (self.wI[l+1][:, :-1].T @ (dφ[l+1] * lS[l+1])) / self.τ
        return lS

    def fit(self, xP, yR):
        xP = np.append(xP, 1)
        η1, η2, η3 = self.ηS

        # Passagem direta (forward) através das camadas ocultas
        curr_in = xP
        new_in, new_uP, new_hP = [], [], []
        for l in range(self.nL):
            u = self.wR[l] @ self.hP[l] + self.wI[l] @ curr_in
            h = self.hP[l] * (1 - 1 / self.τ) + Activation(u, self.act) / self.τ
            new_in.append(curr_in)
            new_uP.append(u)
            new_hP.append(h)
            curr_in = np.append(h, 1)

        yP = self.wO @ new_hP[-1]
        eS = yR - yP

        # Traços de elegibilidade com φ'(u(t)), h(t-1) e entrada no instante t
        dφ = [dActivation(u, self.act) for u in new_uP]
        for l in range(self.nL):
            self.pS[l] = np.outer(dφ[l], self.hP[l]) / self.τ + (1 - 1 / self.τ) * self.pS[l]
            self.qS[l] = np.outer(dφ[l], new_in[l]) / self.τ + (1 - 1 / self.τ) * self.qS[l]

        # Sinais de aprendizado calculados com os pesos anteriores à atualização
        lS = self.LearningSignals(eS, dφ)

        for l in range(self.nL):
            self.wR[l] += η2 * lS[l][:, None] * self.pS[l]
            self.wI[l] += η3 * lS[l][:, None] * self.qS[l]
        self.wO += η1 * np.outer(eS, new_hP[-1])

        if self.store_hist:
            for l in range(self.nL):
                self.wR_hist[l].append(self.wR[l].flatten())
                self.wI_hist[l].append(self.wI[l].flatten())
            self.wO_hist.append(self.wO.flatten())

        # Atualização de estados: pontual e intervalares avançam juntos para h(t)
        # (hL2/hU2 defasados em h(t-1) faziam o intervalo não conter a predição pontual)
        self.hP = [h.copy() for h in new_hP]
        self.Restore()
        self.uP = new_uP
        self.x = xP
        self.k += 1

    def Advance(self, x):
        # Um passo livre a partir do estado de predição hP2 (avança hP2)
        curr_in = np.append(x, 1)
        for l in range(self.nL):
            hP = self.hP2[l]
            uP = (self.wR[l] @ hP) + (self.wI[l] @ curr_in)
            hP = hP * (1 - 1 / self.τ) + Activation(uP, self.act) / self.τ
            self.hP2[l] = hP
            curr_in = np.append(hP, 1)
        return self.wO @ self.hP2[-1]

    def Predict(self, x):
        return self.Advance(x)[self.n]

    def PredictionVector(self, x):
        return self.Advance(x)

    def PredictSingle(self, xP, show=False):
        return self.Advance(xP)[self.n]

    def PredictIntr(self, xP, xL, xU, ep, show=False):
        '''
        Granular one-step prediction with interval weights [w/(1+ep), w(1+ep)] and exact interval arithmetic.
        Interval products take the min/max of the endpoint products; φ (tanh/sigmoid/relu) is non-decreasing and
        the leaky update (1 - 1/τ)h + φ(u)/τ is non-decreasing in h and u (τ >= 1), so the bounds enclose the
        point path at every layer and step: yL <= yP <= yU (inclusion isotonicity).
        '''
        curr_inP = np.append(xP, 1)
        curr_inL = np.append(xL, 1)
        curr_inU = np.append(xU, 1)
        a = 1 - 1 / self.τ

        for l in range(self.nL):
            wR, wI = self.wR[l], self.wI[l]
            hP, hL, hU = self.hP2[l], self.hL2[l], self.hU2[l]
            wRL, wRU = WeightInterval(wR, ep)
            wIL, wIU = WeightInterval(wI, ep)

            uP = (wR @ hP) + (wI @ curr_inP)
            rL, rU = IntervalMatVec(wRL, wRU, hL, hU)
            iL, iU = IntervalMatVec(wIL, wIU, curr_inL, curr_inU)
            uL, uU = rL + iL, rU + iU

            hP = a * hP + Activation(uP, self.act) / self.τ
            hL = a * hL + Activation(uL, self.act) / self.τ
            hU = a * hU + Activation(uU, self.act) / self.τ

            self.hP2[l] = hP
            self.hL2[l] = hL
            self.hU2[l] = hU

            curr_inP = np.append(hP, 1)
            curr_inL = np.append(hL, 1)
            curr_inU = np.append(hU, 1)

        wOL, wOU = WeightInterval(self.wO, ep)
        yP = self.wO @ self.hP2[-1]
        yL, yU = IntervalMatVec(wOL, wOU, self.hL2[-1], self.hU2[-1])

        if show:
            print('-----------------')

        return np.array([yL[self.n], yP[self.n], yU[self.n]])

    def Restore(self):
        self.hP2 = [h.copy() for h in self.hP]
        self.hL2 = [h.copy() for h in self.hP]
        self.hU2 = [h.copy() for h in self.hP]

    def ReturnParameters(self):
        # Cópia profunda: grânulos criados/fundidos a partir deste não compartilham pesos
        params = [self.wR, self.wI, self.pS, self.qS, self.hP, self.hP2, self.hL2, self.hU2, self.wO, self.x, self.uP]
        return copy.deepcopy(params)

    def ReceiveParameters(self, params):
        params = copy.deepcopy(list(params))
        if len(params) == 10:  # formato do eRNN_GD_R1 (sem uP)
            params.append([np.zeros(size) for size in self.nR])
        [self.wR, self.wI, self.pS, self.qS, self.hP, self.hP2, self.hL2, self.hU2, self.wO, self.x, self.uP] = params
