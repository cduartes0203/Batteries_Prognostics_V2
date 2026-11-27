import numpy as np
import math
from Testing.eRNN_R5 import *
from Functions.DataCloud_V5 import *
from Functions.RLS import *
import copy



def _XID(cloud):
    # Identidade de origem de cada amostra do grânulo (grânulos criados antes deste registro: o próprio ID)
    xid = getattr(cloud, 'xid', None)
    if xid is None or len(xid) != len(cloud.x): return [cloud.ID] * len(cloud.x)
    return list(xid)

def RefEoL(sig, eol, kX, strict=False):
    '''
    Referência da RUL real baseada no limiar de EoL (eq. 12): RUL_real(k) = iEoL - k,
    com iEoL = primeira amostra em que o HI <= eol e k o instante de cada amostra (kX, 4ª saída de PrepareData).
    O AutoCloud calcula rulR = ref - self.k, com self.k = j + 2 na iteração j; logo ref = iEoL - kX[0] + 2.
    Retorna:
        ref : referência passada ao AutoCloud
        J   : nº de iterações com RUL real >= 1 (instantes k < iEoL)
        L   : limite da RUL prevista (lim), igual ao nº de amostras a partir de kX[0]
    Se o fluxo de amostras termina antes de iEoL - 1 (nO grande): strict=True -> ValueError; senão, aviso.
    '''
    sig = np.asarray(sig, dtype=float)
    kX = np.asarray(kX, dtype=int)
    if not np.any(sig <= eol): raise ValueError('O HI nunca atinge o limiar de EoL.')
    iEoL = int(np.argmax(sig <= eol))
    if kX[0] >= iEoL: raise ValueError(f'A primeira amostra (k = {kX[0]}) já está no EoL (iEoL = {iEoL}).')
    if kX[-1] < iEoL - 1:
        msg = (f'O fluxo termina em k = {kX[-1]}, antes do EoL (iEoL = {iEoL}): '
               f'a última RUL real avaliada é {iEoL - kX[-1]} (reduza nO).')
        if strict: raise ValueError(msg)
        print('Aviso: ' + msg)
    ref = iEoL - int(kX[0]) + 2
    return ref, int(np.sum(kX < iEoL)), len(sig) - int(kX[0])

class AutoCloud:
	'''
	Granularidade do intervalo de predição do HI pelo protocolo 3 (Al-Hmouz et al., 2024, eq. 10):
	[x_pred - rho*e(x_pred), x_pred + rho*e(x_pred)], com e(.) o erro estimado pelo modelo RLS (UpdateRLS).
	RUL intervalar (RULInterv): limites pela análise intervalar da rede, com o mesmo rho nos intervalos injetados.

	Mudanças em relação ao AutoCloud_V4_6: rho estático OU dinâmico (rho_mode).
		'static'    : rho constante (= rho), como no V4_6 (padrão; resultados idênticos ao V4_6).
		'grid'      : Princípio da Granularidade Justificável online: para cada candidato da grade log-espaçada
		              rho_grid (rho_n valores entre rho_min e rho_max), acumula cobertura e especificidade do intervalo
		              do HI um passo à frente (fator de esquecimento rho_lam; 1.0 = todos os passos com o mesmo peso) e
		              escolhe rho = argmax V(rho), V = média(cov) * média(sp) (empate: menor rho; V = 0 em todos:
		              rho mantido).
		'grad'      : subida de gradiente em v_k(rho) = sigma((rho - s_k)/rho_T) * (1 - 2*rho*e_k/sp_range)
		              (cobertura suavizada * especificidade), com média do gradiente m_k = rho_beta*m_{k-1} +
		              (1 - rho_beta)*g_k e rho <- clip(rho + rho_eta*m_k, rho_min, rho_max) (V4_5/V4_7).
		'conformal' : controle da cobertura (conformal adaptativo): rho <- clip(rho*(1 + rho_kappa*(miss_k - rho_alpha))),
		              miss_k = 1 se o valor real ficou fora do intervalo; leva a cobertura a ~ 1 - rho_alpha.
	Em todos os modos dinâmicos o rho é atualizado em UpdateRLS, depois que o valor real h_{k+1} é conhecido, com o
	escore s_k = |h_{k+1} - x_pred| / e_k e o erro estimado e_k que formou o intervalo do passo (antes do ajuste do RLS):
	o valor real está no intervalo de rho se e somente se s_k <= rho, e a largura do intervalo é 2*rho*e_k.
	Causal: o rho do passo k+1 usa apenas passos até k. O mesmo rho forma o intervalo do HI e os intervalos injetados
	na previsão recursiva da RUL (RULInterv). O rho não altera as predições pontuais (hiP, rulP, WAPE).
	Passos sem erro estimado (e_k = 0, antes do 1º ajuste do RLS) não atualizam o rho.
	restart: o rho e o estado da atualização (acumuladores da grade, gradiente médio) continuam no rolamento seguinte;
	rho_hist / rho_s_hist (rho usado e escore de cada passo) são locais ao rolamento.

	Mudanças em relação ao AutoCloud_V5: treino em malha fechada das RNNs (eRNN_R5, via DataCloud_V5).
	cl_H     : 0 (padrão)  -> sem treino em malha fechada: resultados idênticos ao AutoCloud_V5;
	           > 0         -> após o fit de cada grânulo ativo no passo k, repetição causal de até cl_H passos em
	                          previsão livre (eRNN_R5.ClosedLoopFit): parte de uma janela observada e do estado
	                          recorrente guardados, realimenta as próprias previsões um passo à frente e corrige o erro
	                          em cada passo (alvos h_{t+1} <= h_{k+1}, já conhecidos no passo k).
	cl_eta   : multiplicador das taxas η nas atualizações da repetição.
	cl_every : repetição a cada cl_every passos.
	restart  : esvazia o buffer da repetição de cada RNN (janelas de rolamentos diferentes não são contíguas).
	'''
	def __init__(self,m,nI=1,nR=[1],nO=1,mO=0,ηS=[0.1,0.1,0.1],
			  tau=1,eol=0,ref=0,act='tanh',
			  wtaG=True,wtaP=True,store=True,feedback='symmetric',

			  rho=1.0,intr_state='reset',
			  rls_mode ='log',nf_rls = 3, ff_rls=1, dt_rls = 1e3,
			  rho_mode='static',rho_min=1e-3,rho_max=10.0,sp_range=1.0,
			  rho_n=50,rho_lam=1.0,
			  rho_eta=0.2,rho_T=0.3,rho_beta=0.0,
			  rho_kappa=0.05,rho_alpha=0.1,
			  cl_H=0,cl_eta=1.0,cl_every=1
			  ):

		# treino em malha fechada (eRNN_R5.ClosedLoopFit): cl_H passos de repetição, taxas x cl_eta, a cada cl_every passos
		if int(cl_H) < 0 or not cl_eta >= 0 or int(cl_every) < 1:
			raise ValueError('requer cl_H >= 0, cl_eta >= 0 e cl_every >= 1.')
		self.cl_H, self.cl_eta, self.cl_every = int(cl_H), float(cl_eta), int(cl_every)

		# rho: múltiplo do erro estimado do HI no intervalo de RULInterv (protocolo 3): fixo (rho_mode='static') ou
		# valor inicial da atualização online (rho_mode='grid', 'grad' ou 'conformal'; ver docstring da classe)
		if not rho >= 0: raise ValueError('rho deve ser >= 0.')
		if rho_mode not in ('static', 'grid', 'grad', 'conformal'):
			raise ValueError("rho_mode deve ser 'static', 'grid', 'grad' ou 'conformal'.")
		self.rho_mode = rho_mode
		self.rho = float(rho)
		if rho_mode != 'static':
			if not 0 < rho_min < rho_max: raise ValueError('requer 0 < rho_min < rho_max.')
			if not sp_range > 0: raise ValueError('sp_range deve ser > 0.')
			self.rho = float(np.clip(self.rho, rho_min, rho_max))
		if rho_mode == 'grid':
			if int(rho_n) < 2: raise ValueError('rho_n deve ser >= 2.')
			if not 0 < rho_lam <= 1: raise ValueError('rho_lam deve estar em (0, 1].')
		if rho_mode == 'grad':
			if not (rho_eta > 0 and rho_T > 0 and 0 <= rho_beta < 1):
				raise ValueError('requer rho_eta > 0, rho_T > 0 e 0 <= rho_beta < 1.')
		if rho_mode == 'conformal':
			if not (rho_kappa > 0 and 0 < rho_alpha < 1): raise ValueError('requer rho_kappa > 0 e 0 < rho_alpha < 1.')
		self.rho_min, self.rho_max, self.sp_range = float(rho_min), float(rho_max), float(sp_range)
		self.rho_lam = float(rho_lam)                       # 'grid': fator de esquecimento
		self.rho_eta, self.rho_T, self.rho_beta = float(rho_eta), float(rho_T), float(rho_beta)   # 'grad'
		self.rho_kappa, self.rho_alpha = float(rho_kappa), float(rho_alpha)                      # 'conformal'
		self.rho_grid = np.geomspace(self.rho_min, self.rho_max, int(rho_n)) if rho_mode == 'grid' else None
		self._S_w, self._S_cov, self._S_sp = 0.0, 0.0, 0.0  # 'grid': somas (com esquecimento) de pesos, cov e sp
		self.V_grid = None                                  # 'grid': V de cada candidato no último passo
		self._rho_m = 0.0                                   # 'grad': gradiente médio
		self.rho_hist = np.array([])        # rho usado no intervalo de cada passo (local ao rolamento)
		self.rho_s_hist = np.array([])      # escore s_k = |erro real| / erro estimado de cada passo (NaN sem erro estimado)
		# Estado recorrente intervalar em RULInterv: 'carry' (propagado entre os passos da previsão, limites garantidos
		# para toda a sequência de janelas intervalares) ou 'reset' (reiniciado no estado pontual a cada passo: o intervalo
		# de saída reflete apenas a incerteza da janela de entrada do passo)
		if intr_state not in ('carry', 'reset'):
			raise ValueError("intr_state deve ser 'carry' ou 'reset'.")
		self.intr_state = intr_state
		# k_seg: deslocamento de k no início do rolamento atual (0 no primeiro; atualizado em restart), de modo que
		# RUL real, tempo e pesos do WAPE sejam locais ao rolamento
		self.k_seg = 0
		self.c_retired = []                # grânulos substituídos em fusões (estado no momento da fusão; gráficos)
		self.Vsum = 0.0
		self.act = act
		# Janelas do HI (mesmas de PrepareData): Y_k com nI valores; Z_k com mO passados (incluindo h_k) e nO futuros.
		# Saída das RNNs: nOut = mO + nO valores; a predição um passo à frente h_{k+1} é a coordenada mO.
		self.nI = nI
		self.nR = nR
		self.nO = nO
		self.mO = mO
		self.nOut = nO + mO
		self.sd = 42
		self.feedback = feedback

		if ηS[0] < 1:
			self.N1 = ηS[0]
		elif ηS[0]>=1:
			self.N1 = 1/(10**ηS[0])

		if ηS[1] < 1:
			self.N2 = ηS[1]
		elif ηS[1]>=1:
			self.N2 = 1/(10**ηS[1])

		if ηS[2] < 1:
			self.N3 = ηS[2]
		elif ηS[2]>=1:
			self.N3 = 1/(10**ηS[2])		

		self.tau = tau
		self.eol = eol
		self.store = store
		self.gCreated = 0
		self.c= np.array([])
		self.alfa= np.array([0.0],dtype=float)
		self.listIntersection = np.zeros((1),dtype=int)
		self.matrixIntersection = np.zeros((1,1),dtype=int)
		self.k=1
		self.m = m
		self.cloud_activation = []
		self.DSIs = [np.array([]) for i in range(nI)]

		if self.act == 0:
			self.act = 'tanh'
		elif self.act == 1:
			self.act = 'relu'
		elif self.act == 2:
			self.act = 'sigmoid'

		self.eolX = 0
		self.HI = np.array([])
		self.DSI = np.array([])
		self.eolDSI = 0
		self.HIp = np.array([])
		self.cycleP=np.array([])

		self.t = np.array([])
		self.rulL = np.array([])
		self.rulP = np.array([])
		self.rulU = np.array([])
		self.rulR = np.array([])
		# 1 quando a trajetória (inferior, pontual, superior) não atinge o EoL e a RUL é truncada em lim
		self.rulCapL = np.array([], dtype=int)
		self.rulCapP = np.array([], dtype=int)
		self.rulCapU = np.array([], dtype=int)
		self.ref = ref
		self.lim = 160

		self.hiR = np.array([])
		self.hiP = np.array([])
		self.hiL = np.array([])
		self.hiU = np.array([])

		self.wape_RUL_hist = np.array([])
		self.wape_RUL = 0
		self.eRul_Sum = 0
		self.eRul_Diff = 0

		self.wape_HI_hist = np.array([])
		self.wape_HI = 0
		self.eHI_Sum = 0
		self.eHI_Diff = 0

		self.wape_HI_hist2 = np.array([])
		self.wape_HI2 = 0
		self.eHI_Sum2 = 0
		self.eHI_Diff2 = 0

		self.eP = np.array([])
		self.eR = np.array([])
		self.wtaG = wtaG
		self.wtaP = wtaP

		if rls_mode not in ('log', 'pol'):
			raise ValueError("rls_mode deve ser 'log' ou 'pol'.")
		if rls_mode == 'log':
			self.rls = RLS_LogarithmicRegressor(ff=ff_rls,dt=dt_rls)
		elif rls_mode == 'pol':
			self.rls = RLS(nf=nf_rls,ff=ff_rls,dt=dt_rls)

		self.sp = 0
		self.cov = 0
		self.V = 0
		self.sp_hist = np.array([])
		self.cov_hist = np.array([])
		self.V_hist = np.array([])

		self.gm = None

		self.transfer = False

	def ErrorModelHI(self, y):
		# Erro absoluto estimado de uma predição de HI de valor y (modelo logarítmico RLS de UpdateRLS(mode='HI'));
		# antes do primeiro ajuste o modelo retorna 0 (intervalo degenerado)
		return abs(self.rls.predict(max(y, 1e-9)))

	def UpdateRLS(self,mode='HI'):
		# Valor real do HI do passo conhecido: atualiza o modelo RLS do erro absoluto do HI e a qualidade do intervalo
		# (o antigo modo 'RUL' foi removido: dividia por zero no 1º passo e sobrescrevia sp/cov/V do HI)
		if mode != 'HI': raise ValueError("UpdateRLS: apenas mode='HI' é suportado.")
		x = self.hiP[-1]
		if x <= 0:                        # regressão logarítmica: requer x > 0 (ReLU pode prever exatamente 0)
			x = 1e-9
		eP = self.rls.predict(x)
		eR = np.abs(self.hiP[-1]-self.hiR[-1])
		# rho do intervalo deste passo e, nos modos dinâmicos, atualização com o valor real (erro estimado |eP| é o
		# mesmo que formou o intervalo: o RLS ainda não foi ajustado com este passo)
		self.rho_hist = np.append(self.rho_hist, self.rho)
		self.UpdateRho(eR, abs(eP))
		self.rls.update(x,eR)
		self.HI_Granularity()

		self.eP = np.append(self.eP,eP)
		self.eR = np.append(self.eR,eR)

	def UpdateRho(self, e, ehat):
		'''
		Atualização online do rho (rho_mode != 'static') com o passo cujo valor real acabou de ser conhecido.
		e: erro absoluto real |h_{k+1} - x_pred|; ehat: erro estimado que formou o intervalo [x_pred -/+ rho*ehat].
		'''
		if ehat <= 1e-12 or not np.isfinite(ehat) or not np.isfinite(e):   # modelo do erro ainda não ajustado
			self.rho_s_hist = np.append(self.rho_s_hist, np.nan)
			return
		s = float(e) / float(ehat)                       # no intervalo de rho  <=>  s <= rho
		self.rho_s_hist = np.append(self.rho_s_hist, s)
		if self.rho_mode == 'static':
			return
		if self.rho_mode == 'grid':
			cov = (s <= self.rho_grid).astype(float)
			sp = np.maximum(0.0, 1.0 - 2.0 * self.rho_grid * ehat / self.sp_range)
			self._S_w = self.rho_lam * self._S_w + 1.0
			self._S_cov = self.rho_lam * self._S_cov + cov
			self._S_sp = self.rho_lam * self._S_sp + sp
			self.V_grid = (self._S_cov / self._S_w) * (self._S_sp / self._S_w)     # média(cov) * média(sp)
			i = int(np.argmax(self.V_grid))              # grade crescente: em empate fica o menor rho
			if self.V_grid[i] > 0:
				self.rho = float(self.rho_grid[i])
		elif self.rho_mode == 'grad':
			z = (self.rho - s) / self.rho_T
			sig = 0.5 * (1.0 + np.tanh(0.5 * z))         # sigmoide numericamente estável
			sp_raw = 1.0 - 2.0 * self.rho * ehat / self.sp_range
			g = sig * (1.0 - sig) / self.rho_T * sp_raw - sig * 2.0 * ehat / self.sp_range
			self._rho_m = self.rho_beta * self._rho_m + (1.0 - self.rho_beta) * g
			self.rho = float(np.clip(self.rho + self.rho_eta * self._rho_m, self.rho_min, self.rho_max))
		elif self.rho_mode == 'conformal':
			miss = 1.0 if s > self.rho else 0.0
			self.rho = float(np.clip(self.rho * (1.0 + self.rho_kappa * (miss - self.rho_alpha)), self.rho_min, self.rho_max))

	def HI_Granularity(self):

		hiR,hiL,hiU,hiP = self.hiR,self.hiL,self.hiU,self.hiP

		#Range = np.max(hiR) - np.min(hiR)
		Range = 1 - 0
		cov, sp, Vsum = np.array([]), np.array([]), 0
		
		for hR,hL,hU in zip(hiR,hiL,hiU):
			if hR>= hL and hR<= hU:
				cov = np.append(cov,1)
			else:
				cov = np.append(cov,0)
			spVal = max(0, 1 - (hU-hL)/Range )
			sp = np.append(sp, spVal)
			Vsum +=  cov[-1]*sp[-1]
		self.sp = sp.mean()
		self.cov = cov.mean()
		self.sp_hist = sp
		self.cov_hist = cov
		#self.V = cov.mean() * sp.mean()
		self.V = (cov * sp).mean() 
		self.Vsum = Vsum

	def WAPE_RUL(self):
		yR, yP, t = self.rulR[-1], self.rulP[-1], (self.k - self.k_seg)**2
		self.eRul_Diff = np.abs(yR*t - yP*t) + self.eRul_Diff
		self.eRul_Sum = np.abs(yR*t) + self.eRul_Sum
		self.wape_RUL = self.eRul_Diff/self.eRul_Sum

		if self.store:
			self.wape_RUL_hist = np.append(self.wape_RUL_hist,self.wape_RUL)

	def WAPE_HI(self):
		yR, yP, t = self.hiR[-1], self.hiP[-1], (self.k - self.k_seg)**2
		self.eHI_Diff = np.abs(yR*t - yP*t) + self.eHI_Diff
		self.eHI_Sum = np.abs(yR*t) + self.eHI_Sum
		self.wape_HI = self.eHI_Diff/self.eHI_Sum

		if self.store:
			self.wape_HI_hist = np.append(self.wape_HI_hist,self.wape_HI)

	def PredictWindow(self,y,show=False):
		ws = self.alfa/np.sum(self.alfa)
		p = (np.array([w*cloud.rnn.PredictionVector(y) for cloud,w in zip(self.c,ws)]))
		p = np.sum(p,axis=0)
		return p	

	def	WAPE_HI2(self,y,zR):
		zP = self.PredictWindow(y,show=False)
		if self.k - self.k_seg <= 2: return       # mesmos passos de WAPE_HI e WAPE_RUL
		t = (self.k - self.k_seg)**2
		self.eHI_Diff2 = np.sum(np.abs(zR*t - zP*t)) + self.eHI_Diff2
		self.eHI_Sum2 = np.sum(np.abs(zR*t)) + self.eHI_Sum2
		self.wape_HI2 = self.eHI_Diff2/self.eHI_Sum2

		if self.store:
			self.wape_HI_hist2 = np.append(self.wape_HI_hist2,self.wape_HI2)	
			
	def AdjustCloudMaster(self,t,x):

			clouds = self.c
			points = []
			axes = np.eye(len(self.c[0].mean))
			for cloud in clouds:
				for axis in axes:
					points.append(cloud.mean + axis * cloud.R)
					points.append(cloud.mean - axis * cloud.R)
			points = np.array(points)
	
			p1 = points[np.argmax(np.linalg.norm(points - points[0], axis=1))]
			p2 = points[np.argmax(np.linalg.norm(points - p1, axis=1))]
	
			center = (p1 + p2) / 2.0
			radius = np.linalg.norm(p1 - p2) / 2.0
	
			for p in points:
				dist = np.linalg.norm(p - center)
				if dist > radius:
					new_radius = (radius + dist) / 2.0
					center = center + (p - center) * (new_radius - radius) / dist
					radius = new_radius
	
			for cloud in clouds:
				dist = np.linalg.norm(cloud.mean - center)
				if dist + cloud.R > radius:
					diff = dist + cloud.R - radius
					direction = (center - cloud.mean) / dist if dist > 0 else np.eye(len(center))[0]   # vetor unitário na dimensão do DSI
					center = center - direction * (diff / 2.0)
					radius += diff / 2.0
		
			
			dim = self.gm.dim
			self.gm.mean = center
			self.gm.R = radius
			self.gm.Rmax = radius
			self.gm.V = (math.pi ** (dim / 2)) / math.gamma((dim / 2) + 1) * (radius ** dim)
			self.gm.n = self.k
			if self.store:
				self.gm.t.append(t)
				if self.k > 1:
					self.gm.x.append(x)
					
	def CalculateSpCov(self):
		vec = []
		for cloud in self.c:
			cloud.cov = cloud.n/self.gm.n
			cloud.sp_sum += max(0,1-cloud.V/self.gm.V)
			cloud.sp = cloud.sp_sum/self.gm.n
			vec.append(cloud.sp)

	def CreateCloud(self,x,t,store=False):
		self.gCreated = self.gCreated + 1
		cloud = DataCloud(x,(self.k-1),self.gCreated,self.rho,self.m,nI=self.nI,nR=self.nR,nO=self.nOut,
					ηS=[self.N1,self.N2,self.N3],tau=self.tau,sd=self.sd,act=self.act,feedback=self.feedback,nPast=self.mO)
		self._CfgRNN(cloud)
		if store:
			cloud.t.append(self.k)
		self.c = np.append(self.c,cloud)

	def _CfgRNN(self, cloud):
		# treino em malha fechada da RNN do grânulo (eRNN_R5); cl_H = 0: desligado (igual ao eRNN_R4)
		cloud.rnn.SetClosedLoop(self.cl_H, self.cl_eta)

	def _NextInput(self, x, v):
		# próxima janela de entrada da RNN na repetição em malha fechada: janela deslocada com o HI previsto v
		return np.append(x[1:], v)

	def Adapt(self,y,z, show=False):
		n = self.c[0].rnn.n
		self.HI = np.append(self.HI,z[n])
		self.hiR = np.append(self.hiR,z[n])
		# WAPE do HI com o valor real do mesmo passo (hiP_k vs hiR_k): calculado aqui, após o valor real ser conhecido
		if self.k - self.k_seg > 2:
			self.WAPE_HI()
		cl = self.cl_H > 0 and (self.k - self.k_seg) % self.cl_every == 0
		for i,cloud in enumerate(self.c):
			if self.alfa[i] > 0:
				cloud.rnn.fit(y,z,t=self.k)
				if cl: cloud.rnn.ClosedLoopFit(self._NextInput)   # treino em malha fechada (eRNN_R5)
				if show: print(cloud.ID,'yes')
				
			else:
				if show: print(cloud.ID,'No')

	
	def DefRnnParams(self, cloud):
		if len(self.c)>1:
			params = self.c[-2].rnn.ReturnParameters()
		if len(self.c)==1:
			params = self.c[-1].rnn.ReturnParameters()
		cloud.rnn.ReceiveParameters(params)

	def MergeCloudsRNN(self, c1, c2, c3):
		p1 = c1.rnn.ReturnParameters()
		p2 = c2.rnn.ReturnParameters()
		params = []
		for item1, item2 in zip(p1, p2):
			if isinstance(item1, list):
				# Média elemento a elemento para cada camada l (wR, wI, pS, qS, hP, etc.)
				camadas_media = [(l1 + l2) / 2.0 for l1, l2 in zip(item1, item2)]
				params.append(camadas_media)
			else:
				# Média direta para arrays NumPy simples (wO, x)
				params.append((item1 + item2) / 2.0)

		c3.rnn.ReceiveParameters(params)


	def RULSingle(self,X,lim=None,show=False):
			if lim==None:
				lim = 160
			self.lim = lim
			# Estado de predição (hP2) a partir do estado atual de cada RNN: WAPE_HI2/PredictWindow avança hP2 e Adapt só o
			# restaura nos grânulos ativos (um grânulo reativado partiria de um estado defasado)
			for cloud in self.c:
				cloud.rnn.Restore()
			xP = X.copy()
			rulP = 0
			Capped = 0
			Check = True
			cont = True

			while True:
				if self.wtaP:
					self.alfa = np.where(self.alfa<np.max(self.alfa),0,self.alfa)
				ws = self.alfa/np.sum(self.alfa)

				Pred = [w*c.rnn.PredictSingle(xP,True) for w,c in zip(ws,self.c)]
				yP = sum(Pred)

				if cont:
					self.hiP = np.append(self.hiP,yP)
					self.hiL = np.append(self.hiL,yP)
					self.hiU = np.append(self.hiU,yP)
					cont = False

				xP = np.delete(np.append(xP,yP),0)

				if Check:
					rulP = rulP + 1
				else:
					rulP = rulP

				if rulP > lim:
					Check = False
					rulP = lim
					Capped = 1

				if yP <= self.eol:
					Check = False
					
				else: None


				if not Check: 
					break
				else: None

				
			for cloud in self.c:
				cloud.rnn.Restore()

			self.t = np.append(self.t,self.k - 2 - self.k_seg)
			self.rulR = np.append(self.rulR,self.ref - (self.k - self.k_seg))
			self.rulP = np.append(self.rulP,rulP)
			self.rulL = np.append(self.rulL,rulP)
			self.rulU = np.append(self.rulU,rulP)
			self.rulCapL = np.append(self.rulCapL,Capped)
			self.rulCapP = np.append(self.rulCapP,Capped)
			self.rulCapU = np.append(self.rulCapU,Capped)

			for i,cloud in enumerate(self.c):
				if self.alfa[i] > 0:
					cloud.t2.append(self.k - 2 - self.k_seg)
					cloud.rulR = np.append(cloud.rulR,self.ref - (self.k - self.k_seg))
					cloud.rulP = np.append(cloud.rulP,rulP)
					cloud.rulL.append(rulP)
					cloud.rulM.append(rulP)
					cloud.hiM.append(self.hiP[-1])
					cloud.rulU.append(rulP)

	def RULInterv(self,X,lim=None,show=False):
		'''
		RUL intervalar: previsão recursiva de RULSingle + protocolo 3 (Al-Hmouz et al., 2024, eq. 10) na janela de entrada
		+ análise intervalar da rede (Wang et al., 2024, Lema 1 / Teorema 1) com pesos numéricos.
		A cada passo n:
		  1) janela pontual X_{k+n} -> predição pontual x_{k+n+1} (igual a RULSingle);
		     janela intervalar [X_{k+n}] -> limites [y_inf, y_sup] pela análise intervalar
		     (W+ x_inf - W- x_sup <= W x <= W+ x_sup - W- x_inf em cada camada; estado recorrente conforme intr_state);
		  2) erro estimado do valor previsto: eP_{k+n} = e(x_{k+n+1}) (modelo RLS logarítmico do HI);
		  3) próximas janelas: pontual recebe x_{k+n+1}; intervalar recebe [x_{k+n+1} - rho*eP, x_{k+n+1} + rho*eP]
		     (valores observados permanecem intervalos degenerados; com o avanço, todas as coordenadas viram intervalos);
		  4) RUL: primeiro passo em que y_inf, x_pred e y_sup atingem o EoL -> rulL, rulP, rulU.
		O intervalo do HI do passo (n = 0, janela observada) é [x_pred - rho*eP, x_pred + rho*eP] (eq. 10).

		No modo 'reset' de intr_state o estado intervalar recomeça no estado pontual a cada passo n.
		Como y_inf <= x_pred <= y_sup em todos os passos, rulL <= rulP <= rulU.
		'''
		if lim==None:
			lim = 160
		self.lim = lim
		if self.wtaP:
			self.alfa = np.where(self.alfa<np.max(self.alfa),0,self.alfa)
		ws = self.alfa/np.sum(self.alfa)

		for cloud in self.c:
			cloud.rnn.Restore()
		xP, xL, xU = X.copy(), X.copy(), X.copy()
		Ruls = [0 for i in range(3)]
		CheckRuls = [True for i in range(3)]
		Capped = [0 for i in range(3)]
		first = True

		while True:
			if self.intr_state == 'reset':
				for cloud in self.c:
					cloud.rnn.hL2 = [h.copy() for h in cloud.rnn.hP2]
					cloud.rnn.hU2 = [h.copy() for h in cloud.rnn.hP2]
			# 1) Predição pontual e limites da análise intervalar (ep = 0: pesos numéricos)
			yL, yP, yU = sum(w*c.rnn.PredictIntr(xP,xL,xU,0.0,False) for w,c in zip(ws,self.c))
			yL, yU = min(yL, yP), max(yU, yP)          # remove apenas resíduos de arredondamento
			# 2) Erro estimado do valor previsto (protocolo 3)
			M = self.rho * self.ErrorModelHI(yP)

			if first:
				self.hiP = np.append(self.hiP, yP)
				self.hiL = np.append(self.hiL, yP - M)
				self.hiU = np.append(self.hiU, yP + M)
				first = False

			# 3) Próximas janelas: pontual e intervalar (protocolo 3 no novo valor previsto)
			xP = np.delete(np.append(xP, yP), 0)
			xL = np.delete(np.append(xL, yP - M), 0)
			xU = np.delete(np.append(xU, yP + M), 0)

			# 4) RUL de cada trajetória (inferior, pontual, superior)
			PredVals = (yL, yP, yU)
			for j,_ in enumerate(Ruls):
				if CheckRuls[j]:
					Ruls[j] = Ruls[j] + 1
				if Ruls[j] > lim:
					CheckRuls[j] = False
					Ruls[j] = lim
					Capped[j] = 1
			for j,_ in enumerate(CheckRuls):
				if CheckRuls[j] and PredVals[j] <= self.eol:
					CheckRuls[j] = False
			if sum(CheckRuls)==0:
				break

		for cloud in self.c:
			cloud.rnn.Restore()

		# HI decrescente e y_inf <= x_pred <= y_sup em todos os passos: rulL <= rulP <= rulU por construção
		self.t = np.append(self.t,self.k - 2 - self.k_seg)
		self.rulR = np.append(self.rulR,self.ref - (self.k - self.k_seg))
		self.rulP = np.append(self.rulP,Ruls[1])
		self.rulL = np.append(self.rulL,Ruls[0])
		self.rulU = np.append(self.rulU,Ruls[2])
		self.rulCapL = np.append(self.rulCapL,Capped[0])
		self.rulCapP = np.append(self.rulCapP,Capped[1])
		self.rulCapU = np.append(self.rulCapU,Capped[2])

		for i,cloud in enumerate(self.c):
			if self.alfa[i] > 0:
				cloud.t2.append(self.k - 2 - self.k_seg)
				cloud.rulR = np.append(cloud.rulR,self.ref - (self.k - self.k_seg))
				cloud.rulP = np.append(cloud.rulP,Ruls[1])
				cloud.rulL.append(Ruls[0])
				cloud.rulM.append(Ruls[1])
				cloud.hiM.append(self.hiP[-1])
				cloud.rulU.append(Ruls[2])

	def RUL_Prediction(self,X,mode='single',lim=None,show=False):
		if mode=='single':
			self.RULSingle(X,lim,show)
		elif mode=='interval':
			self.RULInterv(X,lim,show)
		else:
			raise ValueError("Modo inválido. Escolha 'single' ou 'interval'.")
		# WAPE da RUL: rulR e rulP do mesmo passo já disponíveis; o WAPE do HI é calculado em Adapt
		if self.k - self.k_seg > 2:
			self.WAPE_RUL()

	def MergeClouds(self,X):
		i=0
		while(i<len(self.listIntersection)-1):
			merge=False
			j=i+1
			while(j<len(self.listIntersection)):
				if(self.listIntersection[i] == 1 and self.listIntersection[j] == 1):
					self.matrixIntersection[i,j] = self.matrixIntersection[i,j] + 1
				idI = self.c[i].ID
				idJ = self.c[j].ID
				xMinI = self.c[i].xMin
				xMinJ = self.c[j].xMin
				xMaxI = self.c[i].xMax
				xMaxJ = self.c[j].xMax
				meanI = self.c[i].mean
				meanJ = self.c[j].mean
				meantI = self.c[i].meant
				meantJ = self.c[j].meant
				nI = self.c[i].n
				nJ = self.c[j].n
				tipicalityI = self.c[i].tipicality
				tipicalityJ = self.c[j].tipicality
				trackI = self.c[i].track
				trackJ = self.c[j].track
				varianceI = self.c[i].variance
				varianceJ = self.c[j].variance
				rulLI = self.c[i].rulL
				rulLJ = self.c[j].rulL
				rulUI = self.c[i].rulU
				rulUJ = self.c[j].rulU
				rulMI = self.c[i].rulM
				rulMJ = self.c[j].rulM
				nIntersc = self.matrixIntersection[i,j]
				
				if (nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
					
					merge = True
					self.gCreated = self.gCreated + 1
					n = int(nI + nJ - nIntersc)
					mean = ((nI * meanI) + (nJ * meanJ))/(nI + nJ)
					meant = ((nI * meantI) + (nJ * meantJ))/(nI + nJ)
					variance = ((nI - 1) * varianceI + (nJ - 1) * varianceJ)/(nI + nJ - 2)
					tipicality = ((nI*tipicalityI)+(nJ*tipicalityJ))/(nI + nJ)

					newCloud = DataCloud(X,(self.k-1),self.gCreated,self.rho,self.m,nI=self.nI,nR=self.nR,nO=self.nOut,
									ηS=[self.N1,self.N2,self.N3],tau=self.tau,sd=self.sd,act=self.act,feedback=self.feedback,nPast=self.mO)
					self._CfgRNN(newCloud)
					for id in trackI:
						newCloud.track.append(id)
					for id in trackJ:
						newCloud.track.append(id)
						
					newCloud.UpdateDataCloud(n,mean,meant,variance,tipicality,self.k,X)

					x = self.c[i].x + self.c[j].x
					t = self.c[i].t + self.c[j].t
					xid = _XID(self.c[i]) + _XID(self.c[j])
					t2 = self.c[i].t2 + self.c[j].t2

					rulM = rulMI + rulMJ
					rulL = rulLI + rulLJ
					rulU = rulUI + rulUJ
					
					mat = np.array(list(zip(t,x)), dtype=object)
					col = mat[:, 0].astype(int) 
					_, index = np.unique(col, return_index=True)
					result = mat[np.sort(index)]
					t = result[:, 0].tolist()
					x = result[:, 1].tolist()
					xid = [xid[q] for q in np.sort(index)]          # mesma seleção (amostra repetida: origem do 1º grânulo)

					#mat = np.array(list(zip(t2,rulL,rulM,rulU)), dtype=object)
					#col = mat[:, 0].astype(int) 
					#_, index = np.unique(col, return_index=True)
					#result = mat[np.sort(index)]
					#t2   = result[:, 0].tolist()
					#rulL = result[:, 1].tolist()
					#rulM = result[:, 2].tolist()
					#rulU = result[:, 3].tolist()
					if np.linalg.norm(newCloud.mean - self.c[j].xIni) > np.linalg.norm(newCloud.mean - self.c[i].xIni):
						newCloud.xIni = (self.c[j].xIni)
					else: 
						newCloud.xIni = (self.c[i].xIni)

					newCloud.x = x
					newCloud.xMax = 0.5*(self.c[i].xMax + self.c[j].xMax)
					newCloud.xMin = 0.5*(self.c[i].xMin + self.c[j].xMin)
					newCloud.t = t
					newCloud.xid = xid
					for old in (self.c[i], self.c[j]):              # guardados para os gráficos (identidade de origem)
						old.k_retired = self.k
						old.merged_into = newCloud.ID
						self.c_retired.append(old)
					#newCloud.t2 = t2
					#newCloud.rulL = rulL
					#newCloud.rulM = rulM
					#newCloud.rulU = rulU
					#xMin = np.minimum(xMinI, xMinJ)
					#xMax = np.maximum(xMaxI, xMaxJ)
					#newCloud.xMin = xMin
					#newCloud.xMax = xMax	
					newCloud.AdjustBounds(X,self.k)
						
					self.MergeCloudsRNN(self.c[i],self.c[j],newCloud)
					newCloud.merge = f'G{self.gCreated}: G{idI}+G{idJ}'

					self.cloud_activation.append(newCloud)

					self.listIntersection = np.concatenate((self.listIntersection[0 : i], np.array([1]), self.listIntersection[i + 1 : j],self.listIntersection[j + 1 : np.size(self.listIntersection)]),axis=None)
					self.c = np.concatenate((self.c[0 : i ], np.array([newCloud]), self.c[i + 1 : j],self.c[j + 1 : np.size(self.c)]),axis=None)
					# Pesos dos grânulos alinhados a self.c: o grânulo fundido (posição i) recebe a soma das tipicidades de i e j
					self.alfa = np.concatenate((self.alfa[0 : i], np.array([self.alfa[i] + self.alfa[j]]), self.alfa[i + 1 : j],self.alfa[j + 1 : np.size(self.alfa)]),axis=None)
					M0 = self.matrixIntersection
					M1=np.concatenate((M0[0 : i , :],np.zeros((1,len(M0))),M0[i + 1 : j, :],M0[j + 1 : len(M0), :]))
					M1=np.concatenate((M1[:, 0 : i ],np.zeros((len(M1),1)),M1[:, i+1 : j],M1[:, j+1 : len(M0)]),axis=1)
					col = (M0[:, i] + M0[:, j])*(M0[: , i]*M0[:, j] != 0)
					col = np.concatenate((col[0 : j], col[j + 1 : np.size(col)]))
					lin = (M0[i, :]+M0[j, :])*(M0[i, :]*M0[j, :] != 0)
					lin = np.concatenate((lin[ 0 : j], lin[j + 1 : np.size(lin)]))
					M1[:,i]=col
					M1[i,:]=lin
					M1[i, i + 1 : j] = M0[i, i + 1 : j] + M0[i + 1 : j, j].T;   
					self.matrixIntersection = M1

				j += 1
			if(merge): 
				i = 0
			else: i += 1

	def SimGranularRUL(self,J,L,X,Y,Z,mode_rul='interval'):
		# mode_rul: 'interval' (RULInterv: rulL, rulP, rulU e intervalo do HI) ou 'single' (RULSingle: apenas
		# predições pontuais; rulL = rulP = rulU e hiL = hiP = hiU)
		if mode_rul not in ('interval', 'single'):
			raise ValueError("mode_rul deve ser 'interval' ou 'single'.")
		for j,_ in enumerate(X[:J]):
			self.run(X[j])
			self.RUL_Prediction(Y[j],mode=mode_rul,lim=L,show=False)
			self.WAPE_HI2(Y[j],Z[j])
			self.Adapt(Y[j],Z[j])
			self.UpdateRLS(mode='HI')

	def SimGranulation(self,X,J=None):
		'''
		Simula apenas a granulação do DSI (AutoCloud), sem predição de HI/RUL nem treino das RNNs.
		X: DSI, uma amostra por linha. Aceita a saída X de PrepareData (mesmos grânulos de SimGranularRUL)
		   ou o RS bruto (DataFrame ou array: granula todas as amostras).
		J: número de amostras granuladas; None = todas (J de RefEoL reproduz o trecho de SimGranularRUL).
		'''
		X = np.asarray(X, dtype=float)
		if X.ndim == 1: X = X.reshape(-1, 1)
		for x in X[:J]:
			self.run(x)


	def run(self,X):
		self.alfa = np.zeros((np.size(self.c)),dtype=float)
		self.listIntersection = np.zeros((np.size(self.c)),dtype=int)
		if self.k==1 and not self.transfer:
			self.CreateCloud(X,self.k,self.store)
			self.alfa = np.ones(1)
			self.gm = DataCloud(X,(self.k-1),'GM',self.rho,self.m,nI=self.nI,nR=self.nR,nO=self.nOut,
					ηS=[self.N1,self.N2,self.N3],tau=self.tau,sd=self.sd,act=self.act,feedback=self.feedback,nPast=self.mO)
			self._CfgRNN(self.gm)

		elif self.k==2 and not self.transfer:
			self.c[0].AddDataClaud(X,self.k,self.store)
			self.alfa = np.ones(1)
			self.c[0].AdjustBounds(X,self.k)
			
		elif self.k>=3 or self.transfer:
			i=0
			createCloud = True
			params = [[] for l in range(len(self.c))]
			for i,cloud in enumerate(self.c):
				n = cloud.n +1
				m = ((n-1)/n)*cloud.mean + (1/n)*X
				mT = ((n-1)/n) * cloud.meant + (X.dot(X))/n
				v = mT-m.dot(m)
				ecc = ((1/n)+((m-X).T.dot(m-X))/(n*v))
				n_ecc = ((1/n)+((m-X).T.dot(m-X))/(n*v))/2
				n_tip = (1 - (((1/n)+((m-X).T.dot(m-X))/((n*v))-(1e-12))))/(self.k-2)
		
				if(n_ecc<=(self.m**2 +1)/(2*n)):
					
					createCloud= False
					params[i] = [n,m,mT,v,n_tip]
					self.alfa[i] = n_tip
					self.listIntersection[i] = 1
				else:
					self.alfa[i] = 0
					self.listIntersection[i] = 0
				i+=1

			if self.wtaG: 
				self.alfa = np.where(self.alfa<np.max(self.alfa),0,self.alfa)
				self.listIntersection = np.where(self.alfa>0,1,self.alfa)
			
			for i,cloud in enumerate(self.c):
				if self.alfa[i]>0:
					n,m,mT,v,n_tip = params[i]
					cloud.UpdateDataCloud(n,m,mT,v,n_tip,self.k,X,self.store)
					cloud.AdjustBounds(X,self.k)

			if(createCloud):
				self.CreateCloud(X,self.k,self.store)
				self.listIntersection = np.append(self.listIntersection,1)   # novo grânulo: último de self.c
				self.matrixIntersection = np.pad(self.matrixIntersection, ((0,1),(0,1)), 'constant', constant_values=(0)) 
				self.c[-1].R = 0
				
				self.DefRnnParams(self.c[-1])
				self.alfa = np.append(self.alfa,1)
			
			if not self.wtaG: 
				self.MergeClouds(X)

			self.CalculateSpCov()

		self.AdjustCloudMaster(self.k,X)
		self.cycleP = np.append(self.cycleP,self.k-1)
		self.k=self.k+1

	def restart(self,ref,state='keep',x0=None):
		'''
		Prepara o modelo para um novo rolamento: zera RUL, HI, WAPE e registros; mantém grânulos, pesos das RNNs e modelo do erro.
		state: estado recorrente das RNNs (hP, hP2/hL2/hU2, uP) e traços de elegibilidade (pS, qS) de cada grânulo:
			'keep'  : mantidos (estado do fim do rolamento anterior);
			'reset' : estado inicial (0.1) e traços zerados (eRNN.ResetState);
			'fixed' : estado de equilíbrio da RNN para a primeira janela do novo rolamento, x0 = Y[0] de PrepareData
			          (eRNN.SettleState), e traços zerados. Causal: usa só a janela disponível em k = 0.
		'''
		if state not in ('keep', 'reset', 'fixed'):
			raise ValueError("state deve ser 'keep', 'reset' ou 'fixed'.")
		if state == 'fixed' and x0 is None:
			raise ValueError("state='fixed' requer x0 (primeira janela Y[0] do novo rolamento).")
		self.restart_state = state
		self.eolX = 0
		self.HI = np.array([])
		self.DSI = np.array([])
		self.eolDSI = 0
		self.HIp = np.array([])
		self.cycleP=np.array([])

		# k continua global (tipicidade n_tip/(k-2), gm.n, cobertura dos grânulos): zerá-lo faz k-2 = 0 no 2º passo
		# (alfa = inf -> hiP = NaN, que contamina o RLS e os intervalos). Tempo, RUL real e pesos do WAPE ficam locais ao
		# novo rolamento pelo deslocamento k_seg
		self.k_seg = self.k - 1
		self.t = np.array([])
		self.rulL = np.array([])
		self.rulP = np.array([])
		self.rulU = np.array([])
		self.rulR = np.array([])
		# 1 quando a trajetória (inferior, pontual, superior) não atinge o EoL e a RUL é truncada em lim
		self.rulCapL = np.array([], dtype=int)
		self.rulCapP = np.array([], dtype=int)
		self.rulCapU = np.array([], dtype=int)
		self.ref = ref
		self.lim = 160

		self.hiR = np.array([])
		self.hiP = np.array([])
		self.hiL = np.array([])
		self.hiU = np.array([])

		self.wape_RUL_hist = np.array([])
		self.wape_RUL = 0
		self.eRul_Sum = 0
		self.eRul_Diff = 0

		self.wape_HI_hist = np.array([])
		self.wape_HI = 0
		self.eHI_Sum = 0
		self.eHI_Diff = 0

		self.wape_HI_hist2 = np.array([])
		self.wape_HI2 = 0
		self.eHI_Sum2 = 0
		self.eHI_Diff2 = 0
		self.eP = np.array([])               # erros do modelo RLS (estimado e real) do rolamento atual
		self.eR = np.array([])
		self.rho_hist = np.array([])         # rho e escore de cada passo do rolamento atual (o rho e o estado da
		self.rho_s_hist = np.array([])       # atualização online continuam do rolamento anterior)
		self.transfer = True

		for c in self.c:
			c.restart()
			c.rnn.ClearReplay()                # novo rolamento: janelas não contíguas às do anterior
			if state == 'reset':
				c.rnn.ResetState()
			elif state == 'fixed':
				c.rnn.SettleState(np.asarray(x0, dtype=float).ravel())
