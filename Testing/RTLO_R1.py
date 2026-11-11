'''
RFLO original (Murray, 2019): RNN leaky de UMA camada treinada online com traços de elegibilidade e
retroalimentação ALEATÓRIA fixa (B), com a interface usada pelo AutoCloud_V4_5 / DataCloud (mesma do eRNN_R4).

Passo t (x(t) com o bias: [x; 1]):
    u(t) = wR h(t-1) + wI [x(t); 1]
    h(t) = (1 - 1/τ) h(t-1) + φ(u(t))/τ
    y(t) = wO h(t)
Aprendizado (e = y_real - y(t), ℓ = B e, B fixa e aleatória):
    p_ij(t) = (1 - 1/τ) p_ij(t-1) + φ'(u_i(t)) h_j(t-1)/τ        (traço dos pesos recorrentes)
    q_ij(t) = (1 - 1/τ) q_ij(t-1) + φ'(u_i(t)) [x(t); 1]_j/τ     (traço dos pesos de entrada)
    wR += η_rec ℓ_i p_ij ;  wI += η_in ℓ_i q_ij ;  wO += η_out e h(t)^T

Equivalência: com os mesmos argumentos, RTLO(nI, n, nO, ηS, τ, sd, act, nPast) reproduz
eRNN_R4.eRNN(nI, [n], nO, ηS, τ, sd, act=act, feedback='random', nPast=nPast) (mesmas sementes e equações),
mas é uma implementação independente (camada única, sem listas por camada).

Correções em relação à versão anterior (arquivada em OLD/TEDA*.py):
  - traço q com a entrada atual x(t) (antes: x(t-1)); entrada com bias;
  - fit avança também os estados de predição (hP2, hL2, hU2);
  - PredictIntr com aritmética intervalar exata (IntervalMatVec) e estado intervalar propagado;
  - inicialização com gerador local (np.random.default_rng): não altera o estado global do np.random;
  - ReturnParameters/ReceiveParameters/Restore com cópias (grânulos não compartilham arrays);
  - RUL fora da rede (calculada pelo AutoCloud): PredRul/PredRulIntr removidos.
'''
import copy
import numpy as np
from Functions.Utils_RTLO import Activation, dActivation


def XavierUniform(shape, sd):
    # Gerador local (mesmos valores do eRNN_R4 para a mesma semente)
    rng = np.random.default_rng(sd)
    n_in, n_out = shape
    limit = np.sqrt(6 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=shape)


def WeightInterval(W, ep):
    # Grânulo do peso (protocolo 1 de Al-Hmouz et al., 2024, eq. 4): [w(1-ε), w(1+ε)] ordenado, ε ∈ [0, 1]
    if not 0 <= ep <= 1:
        raise ValueError('No protocolo 1, ε deve estar em [0, 1].')
    a, b = (1 - ep) * W, (1 + ep) * W
    return np.minimum(a, b), np.maximum(a, b)


def IntervalMatVec(WL, WU, xL, xU):
    # Produto intervalar exato [WL, WU] @ [xL, xU]: mínimo e máximo dos 4 produtos dos extremos em cada termo
    c = np.stack([WL * xL, WL * xU, WU * xL, WU * xU])
    return c.min(axis=0).sum(axis=1), c.max(axis=0).sum(axis=1)


class RTLO:
    '''
    nI   : nº de entradas (sem o bias);  nR: nº de neurônios recorrentes (int; [n] também é aceito);
    nO   : comprimento da saída (janela Z_k com nPast valores passados + futuros); h_{k+1} = saída[nPast].
    ηS   : [η_out, η_rec, η_in]; valores >= 1 são expoentes N -> 10^-N.
    τ    : constante de tempo da integração leaky (τ >= 1).
    sd   : semente: wI (sd), wR (sd-1), wO (sd-2), B (sd-3).
    act  : 'tanh' | 'relu' | 'sigmoid' (ou 0, 1, 2).
    feedback: somente 'random' (argumento mantido pela interface do DataCloud).
    '''

    def __init__(self, nI, nR, nO, ηS=None, τ=10, sd=42, mode=None, act='tanh',
                 feedback='random', store_hist=False, nPast=0):
        if not isinstance(nR, (int, np.integer)):
            nR = list(nR)
            if len(nR) != 1: raise ValueError('RTLO tem uma única camada recorrente: use nR = n (ou [n]).')
            nR = nR[0]
        if feedback != 'random':
            raise ValueError("RTLO usa apenas retroalimentação aleatória (feedback='random'); "
                             "para 'symmetric' use eRNN_R4.")
        act = {0: 'tanh', 1: 'relu', 2: 'sigmoid'}.get(act, act)
        if not 0 <= int(nPast) < nO: raise ValueError('nPast deve satisfazer 0 <= nPast < nO.')

        self.nR, self.nO = int(nR), nO
        self.nI = nI + 1                     # entrada com bias
        self.nL = 1
        self.act, self.feedback, self.store_hist = act, feedback, store_hist
        self.flw = mode                      # legado (não usado)
        self.n = int(nPast)                  # índice da predição um passo à frente na saída
        self.k = 1
        if ηS is None: ηS = [0.1, 0.1, 0.1]
        ηS = np.array(ηS, dtype=float).ravel()
        if ηS.size != 3: raise ValueError('ηS deve conter 3 taxas: [η_out, η_rec, η_in].')
        self.ηS = np.where(ηS >= 1, 10.0 ** (-ηS), ηS)
        self.τ = τ

        self.x = np.zeros(self.nI)
        self.hP = 0.1 * np.ones(self.nR)     # estado treinado h(t)
        self.Restore()                       # estados de predição (pontual e intervalares)
        self.uP = np.zeros(self.nR)
        self.pS = np.zeros((self.nR, self.nR))
        self.qS = np.zeros((self.nR, self.nI))

        self.wI = XavierUniform([self.nR, self.nI], sd=sd)
        self.wR = XavierUniform([self.nR, self.nR], sd=sd - 1)
        self.wO = XavierUniform([nO, self.nR], sd=sd - 2)
        self.BS = XavierUniform([self.nR, nO], sd=sd - 3)   # retroalimentação aleatória fixa

        self.wR_hist, self.wI_hist, self.wO_hist = [], [], []

    # ------------------------------------------------------------------ treino
    def fit(self, xP, yR):
        xP = np.append(xP, 1)
        η1, η2, η3 = self.ηS
        u = self.wR @ self.hP + self.wI @ xP
        h = self.hP * (1 - 1 / self.τ) + Activation(u, self.act) / self.τ
        e = yR - self.wO @ h

        dφ = dActivation(u, self.act)
        self.pS = np.outer(dφ, self.hP) / self.τ + (1 - 1 / self.τ) * self.pS
        self.qS = np.outer(dφ, xP) / self.τ + (1 - 1 / self.τ) * self.qS

        ℓ = self.BS @ e                      # sinal de aprendizado aleatório (RFLO)
        self.wR += η2 * ℓ[:, None] * self.pS
        self.wI += η3 * ℓ[:, None] * self.qS
        self.wO += η1 * np.outer(e, h)
        if self.store_hist:
            self.wR_hist.append(self.wR.flatten()); self.wI_hist.append(self.wI.flatten()); self.wO_hist.append(self.wO.flatten())

        self.hP = h.copy()
        self.Restore()
        self.uP, self.x = u, xP
        self.k += 1

    def Step(self, xP):
        # Passo direto sem aprendizado (modelo congelado): avança o estado recorrente
        xP = np.append(xP, 1)
        u = self.wR @ self.hP + self.wI @ xP
        self.hP = self.hP * (1 - 1 / self.τ) + Activation(u, self.act) / self.τ
        self.Restore()
        self.uP, self.x = u, xP
        self.k += 1

    def ResetState(self):
        # Reinicia estados e traços (novo rolamento); pesos mantidos
        self.hP = 0.1 * np.ones(self.nR)
        self.Restore()
        self.uP = np.zeros(self.nR)
        self.pS, self.qS = np.zeros_like(self.pS), np.zeros_like(self.qS)
        self.x = np.zeros_like(self.x)

    # ------------------------------------------------------------------ predição
    def Advance(self, x):
        # Um passo livre a partir do estado de predição hP2 (avança hP2); retorna a saída completa
        u = self.wR @ self.hP2 + self.wI @ np.append(x, 1)
        self.hP2 = self.hP2 * (1 - 1 / self.τ) + Activation(u, self.act) / self.τ
        return self.wO @ self.hP2

    def Predict(self, x):
        return self.Advance(x)[self.n]

    def PredictionVector(self, x):
        return self.Advance(x)

    def PredictSingle(self, xP, show=False):
        return self.Advance(xP)[self.n]

    def PredictIntr(self, xP, xL, xU, ep, show=False):
        '''
        Predição granular de um passo: entrada intervalar [xL, xU], pesos intervalares [w(1-ε), w(1+ε)] e
        aritmética intervalar exata. φ é não decrescente e a integração leaky é não decrescente em h e u (τ >= 1),
        logo yL <= yP <= yU (isotonicidade de inclusão). Avança hP2, hL2 e hU2.
        '''
        inP, inL, inU = np.append(xP, 1), np.append(xL, 1), np.append(xU, 1)
        a = 1 - 1 / self.τ
        wRL, wRU = WeightInterval(self.wR, ep)
        wIL, wIU = WeightInterval(self.wI, ep)
        uP = self.wR @ self.hP2 + self.wI @ inP
        rL, rU = IntervalMatVec(wRL, wRU, self.hL2, self.hU2)
        iL, iU = IntervalMatVec(wIL, wIU, inL, inU)
        self.hP2 = a * self.hP2 + Activation(uP, self.act) / self.τ
        self.hL2 = a * self.hL2 + Activation(rL + iL, self.act) / self.τ
        self.hU2 = a * self.hU2 + Activation(rU + iU, self.act) / self.τ
        wOL, wOU = WeightInterval(self.wO, ep)
        yP = self.wO @ self.hP2
        yL, yU = IntervalMatVec(wOL, wOU, self.hL2, self.hU2)
        return np.array([yL[self.n], yP[self.n], yU[self.n]])

    def Restore(self):
        self.hP2 = self.hP.copy()
        self.hL2 = self.hP.copy()
        self.hU2 = self.hP.copy()

    # ------------------------------------------------------------------ parâmetros (grânulos do AutoCloud)
    def ReturnParameters(self):
        # Mesmo formato do eRNN_R4 (listas por camada), com cópia profunda
        params = [[self.wR], [self.wI], [self.pS], [self.qS], [self.hP], [self.hP2], [self.hL2], [self.hU2],
                  self.wO, self.x, [self.uP]]
        return copy.deepcopy(params)

    def ReceiveParameters(self, params):
        params = copy.deepcopy(list(params))
        if len(params) == 10: params.append([np.zeros(self.nR)])
        un = lambda v: v[0] if isinstance(v, list) else v
        (wR, wI, pS, qS, hP, hP2, hL2, hU2, self.wO, self.x, uP) = params
        self.wR, self.wI, self.pS, self.qS = un(wR), un(wI), un(pS), un(qS)
        self.hP, self.hP2, self.hL2, self.hU2, self.uP = un(hP), un(hP2), un(hL2), un(hU2), un(uP)
