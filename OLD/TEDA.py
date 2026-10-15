import numpy as np
import math
from Functions.RLS import *
from Functions.RTLO_R1 import *

def euclidian_dist(x1,x2):
    return np.linalg.norm(x1 - x2,ord=None)

class DataCloud:
	N=0
	def __init__(self,x,ID,nS=1,nI=1,nR=1,nO=1,ηS=[1,1,1],tau=1,decay=1):
		self.ID = ID
		self.track = [ID]
		self.merged = False
		self.merge = None
		self.n=1
		self.mean=x
		self.meant=np.array(x).dot(np.array(x))
		self.variance=0
		self.pertinency=1
		self.tipicality=1e-12
		self.nS = nS
		self.nI = nI
		self.nR = nR
		self.nO = nO
		self.N1 = ηS[0]
		self.N2 = ηS[1]
		self.N3 = ηS[2]
		self.decay = decay
		self.tau = tau
		self.rnn = RTLO(self.nI, self.nR, self.nO,
						[self.N1, self.N2, self.N3], self.tau, self.decay)
		self.x = []
		self.t = []
		self.R = 0
		self.xI = None
		self.xF = None
		self.Rmax = 0
		self.specificity = 0
		self.coverage = 0
		self.cardinality = 1
		self.v = 0
		self.Dmax = -np.inf
		
	def addDataClaud(self,x):
		self.n=2
		self.mean=(self.mean+x)/2
		self.meant=((self.meant)/2) + (x.dot(x))/2
		self.variance=self.meant-self.mean.dot(self.mean)

	def updateDataCloud(self,n,mean,meant,variance,tipicality):
		self.n=n
		self.mean=mean
		self.meant=meant
		self.variance=variance
		self.tipicality=tipicality
		self.cardinality=self.cardinality + 1

	def calc_Rmax(self):
		R1 = euclidian_dist(self.mean, self.xI)
		R2 = euclidian_dist(self.mean, self.xF)
		if R1 > R2: Rmax = R1
		else: Rmax = R2
		if Rmax > self.Rmax: self.Rmax = Rmax
		if self.Rmax > self.R: self.R = self.Rmax
		#self.Rmax = Rmax

	def calc_Dmax(self,xR,xF):
		D1 = euclidian_dist(self.mean, xR)
		D2 = euclidian_dist(self.mean, xF)
		if D1 > D2: Dmax = D1
		else: Dmax = D2
		if Dmax > self.Dmax: self.Dmax = Dmax
		#print(Dmax)

	def calc_sp(self):
		n = len(self.mean)
		vR = ((math.pi**(n/2))/(math.gamma((n/2)+1)))*(self.R**n)
		vMax = ((math.pi**(n/2))/(math.gamma((n/2)+1)))*(self.Dmax**n)
		self.specificity = 1-(vR/vMax)
	
	def calc_cv(self,k):	
		self.coverage = self.cardinality/(k-1)

	def calc_v(self,k):
		self.calc_cv(k)
		self.calc_sp()
		self.v = self.coverage*self.specificity
	
	def calculate_specificity(self,k):
		sum = 0
		for X in self.x[:]:
			dsample = euclidian_dist(self.mean, X)
			sum = sum + dsample/self.dmax
		self.specificity = sum/k	

	def calc_cv(self,k):	
		self.coverage = self.cardinality/(k-1)	

	def calculate_SpCov(self,k):
		self.calc_cv(k)
		self.calculate_specificity(k)
		return self.specificity*self.coverage
		
class AutoCloud:
	def __init__(self,m,nS=1,nI=1,nR=1,nO=1,ηS=[0.1,0.1,0.1],
			  #N1=1,N2=1,N3=1,
			  tau=1,decay=1,eol=0,fator=1,st=0,ep=0.1,wta=False):
		
		self.st = st
		self.nS = nS
		self.nI = nI
		self.nR = nR
		self.nO = nO
		self.N1 = ηS[0]
		self.N2 = ηS[1]
		self.N3 = ηS[2]
		self.tau = tau
		self.eol = eol
		self.decay = decay
		self.fator = fator

		self.g = 1
		self.gCreated = 1
		self.c= np.array([DataCloud(np.array([0]),self.gCreated,self.nS,self.nI,self.nR,self.nO,[self.N1,self.N2,self.N3],self.tau,self.decay)],dtype=DataCloud)
		self.alfa= np.array([0.0],dtype=float)
		self.intersection = np.zeros((1,1),dtype=int)
		self.listIntersection = np.zeros((1),dtype=int)
		self.matrixIntersection = np.zeros((1,1),dtype=int)
		self.relevanceList = np.zeros((1),dtype=int)
		self.classIndex = []
		self.k=1
		self.m = m
		self.cloud_activation = []
		self.cloud_activation2 = []
		self.cloud_activation3 = []
		self.aux = np.array([])
		self.DSIs = [np.array([]) for i in range(nS)]
		self.OffineGrnls = None

		self.eolX = 0
		self.HI = np.array([])
		self.DSI = np.array([])
		self.eolDSI = 0
		self.HIp = np.array([])
		self.cycleP=np.array([])

		self.rulL = np.array([])
		self.rulP = np.array([])
		self.rulU = np.array([])
		self.rulR2 = np.array([])

		self.rls = RLS_LogarithmicRegressor(0.9,10000)

		self.win_all = wta
		self.ChangePoint = []
		self.xR = 0
		self.xF = 0
		self.Dmax = 0
	
	def add_rulR2(self,n):
		self.rulR2 = np.array([n])

	def adapt(self,y,z):
		#print('cont rnn:',self.k)
		if self.k >5 and self.HI[-1] < self.eol and self.eolX==0:
			self.eolX=self.cycleP[-1]-1
		
		#	self.eolDSI=self.DSI[-1]
		wMax = -np.inf
		self.HI = np.append(self.HI,z[-1])
		tS = sum([cloud.tipicality for cloud in self.c])
		wS = np.array([cloud.tipicality for cloud in self.c])/tS
		for w,cloud in zip(wS,self.c):
			if w > wMax:
				wMax = w
				Best_cloud = cloud
		if self.win_all:
			#for i,cloud in enumerate(self.c):
			#	cloud.rnn.adapt(y,z,1)
			Best_cloud.rnn.fit(y,z)
		if not self.win_all:
			for i,cloud in enumerate(self.c):
				cloud.rnn.fit(y,z)

	def predict(self,y):
		#print('gs:',self.g)
		wSum = sum([cloud.tipicality for cloud in self.c])
		ws = np.array([cloud.tipicality/wSum for cloud in self.c]).reshape(-1,1)
		#print(ws)
		p = np.array([cloud.rnn.predict(y) for cloud in self.c])
		p1 = (p*ws)
		p1 = sum([p1[i][-1] for i in range(len(p))])
		#print('p',p)
		if self.win_all:
			p1 = p[np.argmax(ws)][-1]
			#print(p1)
		return p1
	
	def restore_rnn(self):
		for cloud in self.c:
			cloud.rnn.restore()

	def RUL_single(self,X,Scut=None,wta=False):
		if wta: self.win_all = True
		if Scut is None: Scut = 160
		eP,eL,eU=0,0,0
		pP,pU,pL,rulP,rulU,rulL = [0 for i in range(6)]
		xP,xU_max,xU_min,xL_min,xL_max = [X.copy() for i in range(5)]
		pP = self.predict(xP)*self.fator
		eR = np.abs(self.HI[-1]-pP) 
		self.HIp = np.append(self.HIp,pP)
		vL,vU,vP=[],[],[]

		if pP>0:
			self.rls.update(np.abs(pP), eR)
			eP = self.rls.predict(np.abs(pP))
		self.restore_rnn()
		
		while xP[-1]>self.eol:
			pP = self.predict(xP)*self.fator
			#if self.k ==2: print('pP:',pP)
			xP = np.delete(np.append(xP,pP),0)
			rulP=rulP+1
			
			if rulP == Scut: 
				rulP = 0
				break
			vP.append(pP)
		#print(vP) 
		#plt.plot(vP)
		self.restore_rnn()
		self.rulP = np.append(self.rulP,rulP)
		self.rulL = np.append(self.rulL,rulL)
		self.rulU = np.append(self.rulU,rulU)

		return
	
	def MAE(self,start=None,end=None,rulR=None):
		if end is None: end =np.where(self.cycleP == self.eolX)[0][0] + 1 
		if rulR is None: 
			rulR=self.rulR
		y_true, y_pred = rulR[start:end], self.rulP[start:end]
		mae = np.mean(np.abs(y_true - y_pred))
		return mae
	
	def MAPE(self,start=None,end=None,rulR=None):
		if end is None: end =np.where(self.cycleP == self.eolX)[0][0] + 1 
		if rulR is None: 
			rulR=self.rulR
		y_true, y_pred = rulR[start:end], self.rulP[start:end]
		mape = np.mean(np.abs(y_true - y_pred)/y_true)
		return mape

	def RMSE(self,start=None,end=None,rulR=None):
		if end is None: end =np.where(self.cycleP == self.eolX)[0][0] + 1 
		if rulR is None: 
			rulR=self.rulR
		y_true, y_pred = rulR[start:end], self.rulP[start:end]
		rmse = math.sqrt(np.mean(np.abs(y_true - y_pred)**2))
		return rmse
	
	def TimelyWeightedMAPE(self,rult,start=None,end=None,rulR=None,epsilon=1e-10):
		if end is None: end =np.where(self.cycleP == self.eolX)[0][0] + 1 
		if rulR is None: 
			rulR=self.rulR
		x = rult[start:end]
		y_true, y_pred = rulR[start:end], self.rulP[start:end]
		mape = y_true - y_pred
		mape = np.mean(((x**2)*np.abs(mape))/ ((x**2)*y_true))
		return mape
	
	def mergeClouds(self,MergeOffGrnls=True):
		i=0
		while(i<len(self.listIntersection)-1):
			merge=False
			j=i+1
			while(j<len(self.listIntersection)):
				#print("i",i,"j",j,"l",np.size(AutoCloud.listIntersection),"m",np.size(AutoCloud.matrixIntersection),"c",np.size(AutoCloud.c))
				if(self.listIntersection[i] == 1 and self.listIntersection[j] == 1):
					self.matrixIntersection[i,j] = self.matrixIntersection[i,j] + 1;
				#print('erro merge')
				idI = self.c[i].ID
				idJ = self.c[j].ID
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
				winI = self.c[i].rnn.wIN
				winJ = self.c[j].rnn.wIN
				wrecI = self.c[i].rnn.wHS
				wrecJ = self.c[j].rnn.wHS
				woutI = self.c[i].rnn.wOUT
				woutJ = self.c[j].rnn.wOUT
				hiI = self.c[i].rnn.hI
				hiJ = self.c[j].rnn.hI
				#print(self.c[i].ID,np.max(self.c[i].R),self.k)
				#print(self.c[j].ID,np.max(self.c[j].R),self.k)
				#rI = np.max(self.c[i].R)
				#rJ = np.max(self.c[j].R)

				nIntersc = self.matrixIntersection[i,j]

				if (nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
					if not MergeOffGrnls:
						if self.c[i] or self.c[j] in self.OffineGrnls: 
							#print('break')
							break
					rI = np.max(self.c[i].R)
					rJ = np.max(self.c[j].R)
					
					#print(self.c[i].x + self.c[j].x)
					
					if rI >= rJ: radius = rI
					else: radius = rJ
					merge = True
					self.gCreated = self.gCreated + 1
					#update values
					n = nI + nJ - nIntersc
					mean = ((nI * meanI) + (nJ * meanJ))/(nI + nJ)
					#radius = ((nI * rI) + (nJ * rJ))/(nI + nJ)
					meant = ((nI * meantI) + (nJ * meantJ))/(nI + nJ)
					variance = ((nI - 1) * varianceI + (nJ - 1) * varianceJ)/(nI + nJ - 2)
					tipicality = ((nI*tipicalityI)+(nJ*tipicalityJ))/(nI + nJ)
					wIN = ((winI*tipicalityI)+(winJ*tipicalityJ))/(tipicalityI+tipicalityJ)
					wHS = ((wrecI*tipicalityI)+(wrecJ*tipicalityJ))/(tipicalityI+tipicalityJ)
					wOUT = ((woutI*tipicalityI)+(woutJ*tipicalityJ))/(tipicalityI+tipicalityJ)
					hI = ((hiI*tipicalityI)+(hiJ*tipicalityJ))/(tipicalityI+tipicalityJ)

					newCloud = DataCloud(np.array([0]),self.gCreated,self.nS,self.nI,self.nR,self.nO,[self.N1,self.N2,self.N3],self.tau,self.decay)
					for id in trackI:
						newCloud.track.append(id)
					for id in trackJ:
						newCloud.track.append(id)
						
					newCloud.updateDataCloud(n,mean,meant,variance,tipicality)
					newCloud.R = radius
					
					x = self.c[i].x + self.c[j].x
					t = self.c[i].t + self.c[j].t
					
					mat = np.array(list(zip(t,x)), dtype=object)
					col = mat[:, 0].astype(int) 
					_, index = np.unique(col, return_index=True)
					result = mat[np.sort(index)]
					t = result[:, 0].tolist()
					x = result[:, 1].tolist()
					newCloud.x = x
					newCloud.t = t
					

					dist = 0
					if euclidian_dist(self.c[i].xI,self.c[i].xF) > dist:
						dist = euclidian_dist(self.c[i].xI,self.c[i].xF)
						newCloud.xI = self.c[i].xI
						newCloud.xF = self.c[i].xF
					elif euclidian_dist(self.c[j].xI,self.c[j].xF) > dist:
						dist = euclidian_dist(self.c[j].xI,self.c[j].xF)
						newCloud.xI = self.c[j].xI
						newCloud.xF = self.c[j].xF
					elif euclidian_dist(self.c[i].xI,self.c[j].xF) > dist:
						dist = euclidian_dist(self.c[i].xI,self.c[j].xF)
						newCloud.xI = self.c[i].xI
						newCloud.xF = self.c[j].xF
					elif euclidian_dist(self.c[j].xI,self.c[i].xF) > dist:
						dist = euclidian_dist(self.c[j].xI,self.c[i].xF)
						newCloud.xI = self.c[j].xI
						newCloud.xF = self.c[i].xF
					#newCloud.R = np.array(newCloud.R,radius)
					#print(newCloud.rnn.wIN)
					newCloud.rnn.wIN = wIN
					#print(newCloud.rnn.wIN)

					newCloud.rnn.wHS = wHS
					newCloud.rnn.wOUT = wOUT
					newCloud.rnn.hI = hI
					newCloud.merge = f'G{self.gCreated}: G{idI}+G{idJ}'

					self.aux = np.append(self.aux,newCloud.ID)

					#atualizando lista de interseção
					self.listIntersection = np.concatenate((self.listIntersection[0 : i], np.array([1]), self.listIntersection[i + 1 : j],self.listIntersection[j + 1 : np.size(self.listIntersection)]),axis=None)
					#atualizando lista de data clouds 
					self.c = np.concatenate((self.c[0 : i ], np.array([newCloud]), self.c[i + 1 : j],self.c[j + 1 : np.size(self.c)]),axis=None)
					
					#update  intersection matrix
					M0 = self.matrixIntersection
					#Remover linhas 
					M1=np.concatenate((M0[0 : i , :],np.zeros((1,len(M0))),M0[i + 1 : j, :],M0[j + 1 : len(M0), :]))
					#remover colunas
					M1=np.concatenate((M1[:, 0 : i ],np.zeros((len(M1),1)),M1[:, i+1 : j],M1[:, j+1 : len(M0)]),axis=1)
					#calculando nova coluna
					col = (M0[:, i] + M0[:, j])*(M0[: , i]*M0[:, j] != 0)
					col = np.concatenate((col[0 : j], col[j + 1 : np.size(col)]))
					#calculando nova linha
					lin = (M0[i, :]+M0[j, :])*(M0[i, :]*M0[j, :] != 0)
					lin = np.concatenate((lin[ 0 : j], lin[j + 1 : np.size(lin)]))
					#atualizando coluna
					M1[:,i]=col
					#atualizando linha
					M1[i,:]=lin
					M1[i, i + 1 : j] = M0[i, i + 1 : j] + M0[i + 1 : j, j].T;   
					self.matrixIntersection = M1
				j += 1
			if(merge):
				i = 0
			else:
				i += 1

	def run(self,X,MergeGrnls=True,MergeOffGrnls=False,FitOffGrnls=True,OfflineRun=True):
		for i,x in enumerate(X):
			self.DSIs[i] = np.append(self.DSIs[i], X[i])
		self.aux = np.array([])
		self.aux2 = np.array([])
		if self.k == 1: self.xR = X
		self.listIntersection = np.zeros((np.size(self.c)),dtype=int)
		if self.k==1:
			self.c[0] = DataCloud(X,self.gCreated,self.nS,self.nI,
						 self.nR,self.nO,[self.N1,self.N2,self.N3],self.tau,self.decay)
			self.c[0].x.append(X)
			self.c[0].t.append(self.k)
			self.c[0].xI = X
			self.c[0].R = 0
			self.aux = np.append(self.aux,1)
			self.aux2 = np.append(self.aux,self.c[0].track)

		elif self.k==2:
			self.c[0].addDataClaud(X)
			self.c[0].x.append(X)
			self.c[0].t.append(self.k)
			v = self.c[0].variance
			n1 = self.c[0].n
			n2 = self.c[0].n
			R = math.sqrt((((self.m**2)+1)*n2*v/n1)-v)
			self.c[0].R = R
			self.c[0].cardinality = self.c[0].cardinality+1
			self.aux = np.append(self.aux,1)
			self.aux2 = np.append(self.aux,self.c[0].track)

		elif self.k>=3 and MergeGrnls:
			i=0
			createCloud = True
			self.alfa = np.zeros((np.size(self.c)),dtype=float)
			for cloud in self.c:
				n= cloud.n +1
				mean = ((n-1)/n)*cloud.mean + (1/n)*X
				meant = ((n-1)/n) * cloud.meant + (X.dot(X))/n
				variance=meant-mean.dot(mean)
				eccentricity = (1/n)+((mean-X).T.dot(mean-X))/(n*variance)
				typicality = 1 - (eccentricity-(1e-12))
				norm_eccentricity = eccentricity/2
				norm_typicality = typicality/(self.k-2)
				cloud.eccAn = eccentricity
				n1 = n
				n2 = n
				R = math.sqrt((((self.m**2)+1)*n2*variance/n1)-variance)
				if(norm_eccentricity<=(self.m**2 +1)/(2*n)):
					createCloud= False
					if FitOffGrnls and OfflineRun:
						cloud.updateDataCloud(n,mean,meant,variance,typicality)
						self.alfa[i] = norm_typicality
						self.listIntersection.itemset(i,1)
						#aux = np.append(aux,cloud.ID)
						cloud.x.append(X)
						cloud.t.append(self.k)
						cloud.xF = X
						cloud.R = R

					elif not FitOffGrnls and not OfflineRun:
						cloud.x.append(X)
						cloud.t.append(self.k)
						if not cloud in self.OffineGrnls:
							cloud.updateDataCloud(n,mean,meant,variance,typicality)
							self.alfa[i] = norm_typicality
							self.listIntersection.itemset(i,1)
							#aux = np.append(aux,cloud.ID)
							#cloud.x.append(X)
							#cloud.t.append(self.k)
							cloud.xF = X
							cloud.R = R
					self.aux = np.append(self.aux,cloud.ID)
					self.aux2 = np.append(self.aux2,cloud.track)
				else:
					self.alfa[i] = 0
					self.listIntersection.itemset(i,0)
				i+=1

			if(createCloud):
				self.gCreated = self.gCreated + 1
				self.c = np.append(self.c,DataCloud(X,self.gCreated,self.nS,self.nI,
						 self.nR,self.nO,[self.N1,self.N2,self.N3],self.tau,self.decay))
				self.listIntersection = np.insert(self.listIntersection,i,1)
				self.matrixIntersection = np.pad(self.matrixIntersection, ((0,1),(0,1)), 'constant', constant_values=(0)) 
				self.c[-1].x.append(X)
				self.c[-1].t.append(self.k)
				self.c[-1].xI = X
				self.c[-1].R = 0
				self.aux = np.append(self.aux,self.c[-1].ID)
				self.aux2 = np.append(self.aux2,self.c[-1].track)

			self.mergeClouds(MergeOffGrnls)
		
		elif self.k>=3 and not MergeGrnls:
			i=0
			createCloud = True
			self.alfa = np.zeros((np.size(self.c)),dtype=float)
			wcloud = None
			max_tip = -np.inf

			for cloud in self.c:
				n= cloud.n +1
				mean = ((n-1)/n)*cloud.mean + (1/n)*X
				meant = ((n-1)/n) * cloud.meant + (X.dot(X))/n
				variance=meant-mean.dot(mean)
				eccentricity = (1/n)+((mean-X).T.dot(mean-X))/(n*variance)
				typicality = 1 - (eccentricity-(1e-12))
				norm_eccentricity = eccentricity/2
				norm_typicality = typicality/(self.k-2)
				cloud.eccAn = eccentricity
				R = math.sqrt((((self.m**2)+1)*n*variance/n)-variance)

				if(norm_eccentricity<=(self.m**2 +1)/(2*n)):
					createCloud= False
					if norm_typicality > max_tip:
						max_tip = norm_typicality
						parameters = n,mean,meant,variance,typicality
						wcloud = cloud

			if not createCloud:
				n,mean,meant,variance,typicality = parameters
				wcloud.updateDataCloud(n,mean,meant,variance,typicality)
				wcloud.x.append(X)
				wcloud.t.append(self.k)
				wcloud.xF = X
				wcloud.R = R
				self.aux = np.append(self.aux,wcloud.ID)
				self.aux2 = np.append(self.aux2,wcloud.track)

			if createCloud:
				self.gCreated = self.gCreated + 1
				self.c = np.append(self.c,DataCloud(X,self.gCreated,self.nS,self.nI,
						 self.nR,self.nO,[self.N1,self.N2,self.N3],self.tau,self.decay))
				self.c[-1].x.append(X)
				self.c[-1].xI = X
				self.c[-1].R = 0
				self.aux = np.append(self.aux,self.c[-1].ID)
				self.aux2 = np.append(self.aux2,self.c[-1].track)

		if self.Dmax < euclidian_dist(x1=self.xR, x2=X):
			self.Dmax = euclidian_dist(x1=self.xR, x2=X)
			self.xF = X

		if len(self.cycleP) == 0: 
			self.cycleP = np.append(self.cycleP,self.st)
		else: self.cycleP = np.append(self.cycleP,self.cycleP[-1]+1)
		if self.k>1:
			self.rulR2 = np.append(self.rulR2,self.rulR2[-1]-1)

		self.cloud_activation.append(self.aux)
		self.cloud_activation2.append(self.aux)
		self.cloud_activation3.append(self.aux2)
		#self.rulR = np.append(self.rulR, )
		self.k=self.k+1
	
	def reset_rul(self):
		self.eolX = 0
		self.HI  = np.array([1])
		self.cycleP = np.array([])
		self.rulR = np.array([])
		self.rulL = np.array([])
		self.rulU = np.array([])
		self.rulR2 = np.array([])
		self.cloud_activation2 = []
		self.rulP = np.array([])
		#self.OffLine_granules = None
		#self.cycleP = np.append(self.cycleP,self.st)
		#for cloud in self.c:
		#	cloud.rnn.restore_ni2()
