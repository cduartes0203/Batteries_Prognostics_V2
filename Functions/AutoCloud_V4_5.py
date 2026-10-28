import numpy as np
import math
from Testing.eRNN_R4 import *
from OLD.DataCloud_V4 import *
from Functions.RLS import *
import copy


class AutoCloud:
	'''
	rho dinâmico (granularidade do intervalo de predição), ajustado a cada iteração pelo Princípio da
	Granularidade Justificável em modo online:
		- rho começa em 0; a predição do passo k usa o rho escolhido com os passos anteriores (causal);
		- para cada candidato da grade rho_grid (log-espaçada entre rho_min e rho_max, n_rho valores),
		  o intervalo do passo k é calculado e, quando o valor real é conhecido, acumulam-se cobertura e
		  especificidade com fator de esquecimento lam (1.0 = todos os passos com o mesmo peso);
		- V(rho) = média(cov) * média(sp) (abordagem do artigo); rho_{k+1} = argmax V (empate: menor rho);
		  se nenhum candidato cobriu ainda (V = 0), rho é mantido;
		- V é o valor máximo do passo e Vsum o acumulado de V.
	rho_target:
		'HI' : intervalo do HI um passo à frente; valor real conhecido no passo seguinte (causal); Range = 1.
		'RUL': intervalo da RUL; usa a RUL real (ref - k), que depende do EoL conhecido só ao fim da vida
		       (modo offline/oráculo, para análise); Range = lim. Custo: n_rho previsões completas por passo.
	'''
	def __init__(self,m,nI=1,nR=[1],nO=1,ηS=[0.1,0.1,0.1],
			  tau=1,eol=0,ref=0,mode='past',act='tanh',st=0,end=0,
			  wtaG=True,wtaP=True,store=True,feedback='symmetric',mO=0,
			  rho=1e-3,lam=1.0,intr_state='reset',
			  rho_dyn=True,rho_eta=0.2,rho_T=0.3,rho_beta=0.0,rho_min=1e-3,rho_max=10.0,sp_range=1.0,
			  rls_mode ='log',nf_rls = 3, ff_rls=1, dt_rls = 1e3
			  ):

		# rho: múltiplo do erro estimado do HI no intervalo de RULInterv2 (protocolo 3); valor inicial se rho_dyn=True
		self.rho = rho
		# rho dinâmico (sugestão 3): subida de gradiente online em V = média(cov*sp) do HI.
		# A cada passo k, conhecido o erro real |e_k| e o erro estimado e_k usado no intervalo, com escore s_k = |e_k|/e_k:
		#   v_k(rho) = sigma((rho - s_k)/T) * (1 - 2*rho*e_k/R)        (cobertura suavizada * especificidade)
		#   g_k = sigma'(.)/T * (1 - 2*rho*e_k/R) - sigma(.) * 2*e_k/R
		#   m_k = beta*m_{k-1} + (1-beta)*g_k  (média do gradiente com esquecimento);  rho <- clip(rho + eta*m_k)
		# Causal: o rho do passo k+1 usa apenas passos até k. T é adimensional (escala do escore).
		self.rho_dyn = rho_dyn
		self.rho_eta, self.rho_T, self.rho_beta = rho_eta, rho_T, rho_beta
		self.rho_min, self.rho_max, self.sp_range = rho_min, rho_max, sp_range
		self._rho_m = 0.0                   # gradiente médio (momento)
		self.ehat_last = None               # erro estimado usado no intervalo do HI do passo atual
		self.rho_used = np.array([])        # rho usado em cada passo
		self.rho_grad_hist = np.array([])
		# Estado recorrente intervalar em RULInterv2: 'carry' (propagado entre os passos da previsão, limites garantidos
		# para toda a sequência de janelas intervalares) ou 'reset' (reiniciado no estado pontual a cada passo: o intervalo
		# de saída reflete apenas a incerteza da janela de entrada do passo)
		if intr_state not in ('carry', 'reset'):
			raise ValueError("intr_state deve ser 'carry' ou 'reset'.")
		self.intr_state = intr_state
		self.lam = lam
		self.S_w = 0.0                      # soma (com esquecimento) dos pesos dos passos
		self.Vsum = 0.0
		self.rho_hist = np.array([])        # rho usado na predição de cada passo
		self.V_hist = np.array([])          # V após cada atualização
		self.Vsum_hist = np.array([])
		self._pending = None                # intervalos dos candidatos aguardando o valor real
		self.vSum = 0
		self.st = st
		self.mode = mode
		self.act = act
		self.end = end
		self.nI = nI
		self.nR = nR
		self.nO = nO
		self.sd = 42
		self.feedback = feedback
		self.mO = mO

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
		self.y = None
		self.g = 1
		self.gCreated = 0
		self.c= np.array([])
		self.alfa= np.array([0.0],dtype=float)
		self.intersection = np.zeros((1,1),dtype=int)
		self.listIntersection = np.zeros((1),dtype=int)
		self.matrixIntersection = np.zeros((1,1),dtype=int)
		self.relevanceList = np.zeros((1),dtype=int)
		self.classIndex = []
		self.k=1
		self.m = m
		self.cloud_activation = []
		self.aux = np.array([])
		self.DSIs = [np.array([]) for i in range(nI)]
		self.OffineGrnls = None

		if self.act == 0:
			self.act = 'tanh'
		elif self.act == 1:
			self.act = 'relu'
		elif self.act == 2:
			self.act = 'sigmoid'

		if self.mode == 0:
			self.mode = 'past'
		elif self.mode == 1:
			self.mode = 'ahead'
			
		self.eolX = 0
		self.HI = np.array([])
		self.DSI = np.array([])
		self.eolDSI = 0
		self.HIp = np.array([])
		self.cycleP=np.array([])

		self.t = np.array([])
		self.rulL = np.array([])
		self.rulP = np.array([])
		self.rulP2 = np.array([])
		self.rulU = np.array([])
		self.rulR = np.array([])
		# 1 quando a trajetória (inferior, pontual, superior) não atinge o EoL e a RUL é truncada em lim
		self.rulCapL = np.array([], dtype=int)
		self.rulCapP = np.array([], dtype=int)
		self.rulCapU = np.array([], dtype=int)
		self.ref = ref
		self.rulR2 = np.array([])
		self.rulR_Train = np.array([])
		self.lim = 160
		self.yP = np.array([])
		self.yR = np.array([])

		self.hiR = np.array([])
		self.hiP = np.array([])
		self.hiL = np.array([])
		self.hiU = np.array([])

		self.wMape_hi = 0
		self.wMape_rul = 0

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
		self.xR = 0
		self.xF = 0
		self.Dmax = 0

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

		self.dim = None
		self.gm = None
		self.mean_gm = 0
		self.meant_gm = 0
		self.var_gm = 0
		self.xMax_gm = None
		self.Rmax_gm = 0
		self.Vmax_gm = 0

	def UpdateRLS(self,mode='RUL'):

		if mode == 'RUL':
			#x = self.rulP[-1]
			#eP = self.rls.predict(x)
			#eR = np.abs(self.rulP[-1]-self.rulR[-1])
			#self.rls.update(x,eR)
			#self.RUL_Granularity()

			x = self.hiP[-1]
			if x < 0:
				x = 1e-9
			eP = self.rls.predict(x)
			eR = np.abs(self.hiP[-1]-self.hiR[-1])
			self.rls.update(x,eR)
			self.RUL_Granularity()
			# Valor real do HI conhecido: atualiza o rho do próximo passo
			if self.rho_dyn and self.ehat_last is not None:
				self.UpdateRhoGrad(eR, self.ehat_last)
				self.ehat_last = None

		if mode == 'HI':
			x = self.hiP[-1]
			if x < 0:
				x = 1e-9
			eP = self.rls.predict(x)
			eR = np.abs(self.hiP[-1]-self.hiR[-1])
			self.rls.update(x,eR)
			self.HI_Granularity()
			# Valor real do HI conhecido: atualiza o rho do próximo passo
			if self.rho_dyn and self.ehat_last is not None:
				self.UpdateRhoGrad(eR, self.ehat_last)
				self.ehat_last = None

		self.eP = np.append(self.eP,eP)
		self.eR = np.append(self.eR,eR)

	def UpdateRhoGrad(self, e, ehat):
		# Subida de gradiente online em v_k(rho) = sigma((rho - s_k)/T) * (1 - 2*rho*ehat/R), s_k = |e_k|/ehat
		if ehat <= 0:                                   # modelo de erro ainda não treinado: rho mantido
			self.rho_grad_hist = np.append(self.rho_grad_hist, 0.0)
			return
		s = abs(e) / ehat
		z = (self.rho - s) / self.rho_T
		sig = 0.5 * (1.0 + np.tanh(0.5 * z))            # sigmoide numericamente estável
		sp_raw = 1.0 - 2.0 * self.rho * ehat / self.sp_range
		g = sig * (1.0 - sig) / self.rho_T * sp_raw - sig * 2.0 * ehat / self.sp_range
		self._rho_m = self.rho_beta * self._rho_m + (1.0 - self.rho_beta) * g
		self.rho = float(np.clip(self.rho + self.rho_eta * self._rho_m, self.rho_min, self.rho_max))
		self.rho_grad_hist = np.append(self.rho_grad_hist, g)

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

	def RUL_Granularity(self):
		rulR,rulL,rulP,rulU = self.rulR,self.rulL,self.rulP,self.rulU
		Range = np.max(rulR) - np.min(rulR)
		cov, sp, Vsum = np.array([]), np.array([]), 0

		for rR,rL,rU in zip(rulR,rulL,rulU):
			if rR>= rL and rR<= rU:
				cov = np.append(cov,1)
			else:
				cov = np.append(cov,0)
			spVal = max(0, 1 - (rU-rL)/Range )
			sp = np.append(sp, spVal)
			Vsum +=  cov[-1]*sp[-1]

			self.sp = sp.mean()
			self.cov = cov.mean()
			self.sp_hist = sp
			self.cov_hist = cov
			#self.V = cov.mean() * sp.mean()
			self.V = (cov * sp).mean() 
			self.Vsum = Vsum

	def UpdateRho(self, truth):
		# Atualização online do rho: acumula cov/sp de cada candidato com o valor real e escolhe o melhor V
		target, L, U = self._pending[:3]
		self._pending = None
		Range = 1.0 if target == 'HI' else float(self.lim)
		cov = ((L <= truth) & (truth <= U)).astype(float)
		sp = np.maximum(0.0, 1.0 - (U - L) / Range)
		self.S_w = self.lam * self.S_w + 1.0
		self.S_cov = self.lam * self.S_cov + cov
		self.S_sp = self.lam * self.S_sp + sp
		self.V_grid = (self.S_cov / self.S_w) * (self.S_sp / self.S_w)
		i = int(np.argmax(self.V_grid))          # grade crescente: em empate fica o menor rho
		if self.V_grid[i] > 0:
			self.rho = float(self.rho_grid[i])
		self.V = float(self.V_grid[i])
		self.Vsum += self.V
		self.V_hist = np.append(self.V_hist, self.V)
		self.Vsum_hist = np.append(self.Vsum_hist, self.Vsum)

	def WAPE_RUL(self):
		yR, yP, t = self.rulR[-1], self.rulP[-1], self.k**2
		self.eRul_Diff = np.abs(yR*t - yP*t) + self.eRul_Diff
		self.eRul_Sum = np.abs(yR*t) + self.eRul_Sum
		self.wape_RUL = self.eRul_Diff/self.eRul_Sum

		if self.store:
			self.wape_RUL_hist = np.append(self.wape_RUL_hist,self.wape_RUL)

	def WAPE_HI(self):
		yR, yP, t = self.hiR[-1], self.hiP[-1], self.k**2
		self.eHI_Diff = np.abs(yR*t - yP*t) + self.eHI_Diff
		self.eHI_Sum = np.abs(yR*t) + self.eHI_Sum
		self.wape_HI = self.eHI_Diff/self.eHI_Sum

		if self.store:
			self.wape_HI_hist = np.append(self.wape_HI_hist,self.wape_HI)

	def teda_WAPE(self):
		self.WAPE_RUL()
		self.WAPE_HI()

	def PredictWindow(self,y,show=False):
		ws = self.alfa/np.sum(self.alfa)
		p = (np.array([w*cloud.rnn.PredictionVector(y) for cloud,w in zip(self.c,ws)]))
		p = np.sum(p,axis=0)
		return p	

	def	WAPE_HI2(self,y,zR):
		zP = self.PredictWindow(y,show=False)
		t = self.k**2
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
					direction = (center - cloud.mean) / dist if dist > 0 else np.array([1, 0, 0])
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
		cloud = DataCloud(x,(self.k-1),self.gCreated,self.rho,self.m,self.nI,self.nR,self.nO,
					[self.N1,self.N2,self.N3],self.tau,self.sd,self.mode, self.act,feedback=self.feedback,nPast=self.mO)
		if store:
			cloud.t.append(self.k)
		self.c = np.append(self.c,cloud)

	def CloudOtimization(self):
		vec = np.array([])
		for cloud in self.c:
			cloud.sp = cloud.sp/(self.k-1)
			cloud.cov = cloud.cov/(self.k-1)
			cloud.v = cloud.sp * cloud.cov
			vec = np.append(vec, cloud.v)
		self.vSum = np.sum(vec)

	def Adapt(self,y,z, show=False):
		n = self.c[0].rnn.n
		self.HI = np.append(self.HI,z[n])
		self.hiR = np.append(self.hiR,z[n])
		# Valor real do passo conhecido: atualiza o rho que será usado na próxima predição
		#if self._pending is not None:
		#	self.UpdateRho(z[n] if self._pending[0] == 'HI' else self._pending[3])
		for i,cloud in enumerate(self.c):
			if self.alfa[i] > 0:
				cloud.rnn.fit(y,z)
				if show: print(cloud.ID,'yes')
				
			else:
				if show: print(cloud.ID,'No')

	
	def Predict(self,y,show=False):
		ws = self.alfa/np.sum(self.alfa)
		p = (np.array([w*cloud.rnn.Predict(y) for cloud,w in zip(self.c,ws)]))
		p = np.sum(p)
		return p

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
			xP = X.copy()
			predict = True
			rulP = 0
			Check = True
			PredVal = np.zeros(1)
			cont = True

			while predict:
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
				else: None

				if yP <= self.eol:
					Check = False
					
				else: None


				if not Check: 
					break
				else: None

				
			for cloud in self.c:
				cloud.rnn.Restore()

			self.t = np.append(self.t,self.k - 2)
			self.rulR = np.append(self.rulR,self.ref - self.k)
			self.rulP = np.append(self.rulP,rulP)
			self.rulL = np.append(self.rulL,rulP)
			self.rulU = np.append(self.rulU,rulP)

			for i,cloud in enumerate(self.c):
				if self.alfa[i] > 0:
					cloud.t2.append(self.k-2)
					cloud.rulR = np.append(cloud.rulR,self.ref - self.k)
					cloud.rulP = np.append(cloud.rulP,rulP)
					cloud.rulL.append(rulP)
					cloud.rulM.append(rulP)
					cloud.rulU.append(rulP)


	def RULInterv(self,X,lim=None,show=False):
		if lim==None:
			lim = 160
		self.lim = lim
		if self.wtaG==False and self.wtaP==True:
			self.alfa = np.where(self.alfa<np.max(self.alfa),0,self.alfa)
		ws = self.alfa/np.sum(self.alfa)

		# Predição do passo com o rho atual (escolhido com os passos anteriores)
		Ruls, Capped, (yL, yP, yU) = self.ForecastInterval(X, self.rho, lim, ws)
		self.rho_hist = np.append(self.rho_hist, self.rho)
		self.hiP = np.append(self.hiP,yP)
		self.hiL = np.append(self.hiL,yL)
		self.hiU = np.append(self.hiU,yU)

		ruls = np.array([self.ForecastInterval(X, self.rho, lim, ws)[0] for r in self.rho_grid])
		self._pending = ('RUL', ruls[:, 0], ruls[:, 2], self.ref - self.k)

		# Sem ajuste de Ruls[2]: com aritmética intervalar exata (eRNN_GD_R3) yL <= yP <= yU em todo o horizonte,
		# logo rulL <= rulP <= rulU por construção (HI decrescente: a trajetória inferior atinge o EoL primeiro)
		self.t = np.append(self.t,self.k - 2)
		self.rulR = np.append(self.rulR,self.ref - self.k)
		self.rulP = np.append(self.rulP,Ruls[1])
		self.rulL = np.append(self.rulL,Ruls[0])
		self.rulU = np.append(self.rulU,Ruls[2])
		self.rulCapL = np.append(self.rulCapL,Capped[0])
		self.rulCapP = np.append(self.rulCapP,Capped[1])
		self.rulCapU = np.append(self.rulCapU,Capped[2])

		for i,cloud in enumerate(self.c):
			if self.alfa[i] > 0:
				cloud.t2.append(self.k-2)
				cloud.rulR = np.append(cloud.rulR,self.ref - self.k)
				cloud.rulP = np.append(cloud.rulP,Ruls[1])
				cloud.rulL.append(Ruls[0])
				cloud.rulM.append(Ruls[1])
				cloud.rulU.append(Ruls[2])

	def ErrorModelHI(self, y):
		# Erro absoluto estimado de uma predição de HI de valor y (modelo logarítmico RLS de UpdateRLS(mode='HI'));
		# antes do primeiro ajuste o modelo retorna 0 (intervalo degenerado)
		return abs(self.rls.predict(max(y, 1e-9)))

	def RULInterv2(self,X,lim=None,show=False):
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
			# 2) Erro estimado do valor previsto
			M = self.rho * self.ErrorModelHI(yP)

			if first:
				self.ehat_last = self.ErrorModelHI(yP)      # erro estimado do intervalo do HI deste passo
				self.rho_used = np.append(self.rho_used, self.rho)
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
				if PredVals[j] <= self.eol:
					CheckRuls[j] = False
			if sum(CheckRuls)==0:
				break

		for cloud in self.c:
			cloud.rnn.Restore()

		# HI decrescente e y_inf <= x_pred <= y_sup em todos os passos: rulL <= rulP <= rulU por construção
		self.t = np.append(self.t,self.k - 2)
		self.rulR = np.append(self.rulR,self.ref - self.k)
		self.rulP = np.append(self.rulP,Ruls[1])
		self.rulL = np.append(self.rulL,Ruls[0])
		self.rulU = np.append(self.rulU,Ruls[2])
		self.rulCapL = np.append(self.rulCapL,Capped[0])
		self.rulCapP = np.append(self.rulCapP,Capped[1])
		self.rulCapU = np.append(self.rulCapU,Capped[2])

		for i,cloud in enumerate(self.c):
			if self.alfa[i] > 0:
				cloud.t2.append(self.k-2)
				cloud.rulR = np.append(cloud.rulR,self.ref - self.k)
				cloud.rulP = np.append(cloud.rulP,Ruls[1])
				cloud.rulL.append(Ruls[0])
				cloud.rulM.append(Ruls[1])
				cloud.rulU.append(Ruls[2])

	def ForecastInterval(self, X, rho, lim, ws, first_only=False):
		'''
		Previsão recursiva (pontual e intervalar) com granularidade rho, sem efeitos colaterais.
		Retorna Ruls [L, P, U], Capped [L, P, U] (1 = truncada em lim) e (yL, yP, yU) do primeiro passo.
		Todas as trajetórias partem do estado atual h(t) de cada grânulo (garante yL <= yP <= yU).
		'''
		for cloud in self.c:
			cloud.rnn.Restore()
		xP,xL,xU = X.copy(),X.copy(),X.copy()
		Ruls = [0 for i in range(3)]
		CheckRuls = [True for i in range(3)]
		Capped = [0 for i in range(3)]
		first = None

		while True:
			Pred = [w*c.rnn.PredictIntr(xP,xL,xU,rho,False) for w,c in zip(ws,self.c)]
			PredVals = sum(Pred)
			if rho == 0:
				PredVals = np.array([PredVals[1], PredVals[1], PredVals[1]])
			yL,yP,yU = PredVals
			if first is None:
				first = (yL, yP, yU)
				if first_only:
					break

			xP = np.delete(np.append(xP,yP),0)
			xL = np.delete(np.append(xL,yL),0)
			xU = np.delete(np.append(xU,yU),0)

			for j,_ in enumerate(Ruls):
				if CheckRuls[j]:
					Ruls[j] = Ruls[j] + 1

				if Ruls[j] > lim:
					CheckRuls[j] = False
					Ruls[j] = lim
					Capped[j] = 1

			for j,_ in enumerate(CheckRuls):
				if PredVals[j] <= self.eol:
					CheckRuls[j] = False

			if sum(CheckRuls)==0:
				break

		for cloud in self.c:
			cloud.rnn.Restore()
		return Ruls, Capped, first

	def RUL_Prediction(self,X,mode='single',lim=None,show=False):
		if mode=='single':
			self.RULSingle(X,lim,show)
		elif mode=='interval':
			self.RULInterv2(X,lim,show)
		else:
			raise ValueError("Modo inválido. Escolha 'single' ou 'interval'.")
		if self.k >2:
			self.teda_WAPE()

	def AddRUL(self):
		for cloud in self.cloud_activation:
			cloud.rulP = np.append(cloud.rulP,self.rulP[-1])
		self.cloud_activation = []
	
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

					newCloud = DataCloud(X,self.k-1,self.gCreated,self.rho,self.m,self.nI,self.nR,self.nO,
									[self.N1,self.N2,self.N3],self.tau,self.sd,self.mode, self.act,feedback=self.feedback,nPast=self.mO)
					for id in trackI:
						newCloud.track.append(id)
					for id in trackJ:
						newCloud.track.append(id)
						
					newCloud.UpdateDataCloud(n,mean,meant,variance,tipicality,self.k,X)

					x = self.c[i].x + self.c[j].x
					t = self.c[i].t + self.c[j].t
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
					self.aux = np.append(self.aux,newCloud.ID)

					self.listIntersection = np.concatenate((self.listIntersection[0 : i], np.array([1]), self.listIntersection[i + 1 : j],self.listIntersection[j + 1 : np.size(self.listIntersection)]),axis=None)
					self.c = np.concatenate((self.c[0 : i ], np.array([newCloud]), self.c[i + 1 : j],self.c[j + 1 : np.size(self.c)]),axis=None)
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

	def run(self,X):
		self.alfa = np.zeros((np.size(self.c)),dtype=float)
		self.listIntersection = np.zeros((np.size(self.c)),dtype=int)
		if self.k==1:
			self.xR = X
			self.CreateCloud(X,self.k,self.store)
			self.alfa = np.ones(1)
			self.gm = DataCloud(X,(self.k-1),'GM',self.rho,self.m,self.nI,self.nR,self.nO,
					[self.N1,self.N2,self.N3],self.tau,self.sd,self.mode, self.act,feedback=self.feedback,nPast=self.mO)

		elif self.k==2:
			self.c[0].AddDataClaud(X,self.k,self.store)
			v = self.c[0].variance
			n = self.c[0].n
			self.alfa = np.ones(1)
			self.c[0].AdjustBounds(X,self.k)
			
		elif self.k>=3:
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
				self.CreateCloud(X,self.store)
				self.listIntersection = np.insert(self.listIntersection,i,1)
				self.matrixIntersection = np.pad(self.matrixIntersection, ((0,1),(0,1)), 'constant', constant_values=(0)) 
				self.c[-1].t.append(self.k)
				self.c[-1].R = 0
				
				self.DefRnnParams(self.c[-1])
				self.alfa = np.append(self.alfa,1)
			
			if not self.wtaG: 
				self.MergeClouds(X)

			self.CalculateSpCov()

		self.AdjustCloudMaster(self.k,X)
		self.cycleP = np.append(self.cycleP,self.st+self.k-1)
		self.rulR_Train = np.append(self.rulR_Train,self.end-self.k)
		self.k=self.k+1