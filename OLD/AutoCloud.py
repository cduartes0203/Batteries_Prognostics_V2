import numpy as np
import math
import copy
from OLD.Cloud import *
from OLD.RTLO import *
from Functions.RLS import *

class AutoCloud:
	def __init__(self,m,nS=1,nI=1,nR=1,nO=1,ηS=[0.1,0.1,0.1],
			  tau=1,decay=1,eol=0,fator=1,st=0,end=0,ep=0.1,wta=False):
		
		self.ep = ep
		self.st = st
		self.end = end
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
		self.rulR = np.array([])
		self.rulR_Train = np.array([])
		self.lim = 160

		self.hiR = np.array([])
		self.hiP = np.array([])

		self.rls = RLS_LogarithmicRegressor(0.9,10000)
		self.εP = []
		self.εR = []
		self.win_all = wta
		self.ChangePoint = []
		self.xI = 0
		self.xF = 0
		self.mean = np.zeros(nS)
		self.meant = 0
		self.variance = 0
		self.Dmax = 0
		self.Dvec = np.array([])
	
	def CreateCloud(self,x):
		self.gCreated = self.gCreated + 1
		cloud = DataCloud(x,self.gCreated,self.nI,self.nR,self.nO,
					[self.N1,self.N2,self.N3],self.tau,self.decay)
		cloud.m = self.m
		self.c = np.append(self.c,cloud)

	def add_rulR2(self,n):
		self.rulR2 = np.array([n])

	def adapt(self,y,z):
		self.HI = np.append(self.HI,z[-1])

		if self.k >5 and self.HI[-1] < self.eol and self.eolX==0:
			self.eolX=self.cycleP[-1]
		wMax = -np.inf
		tS = sum([cloud.tipicality for cloud in self.c])
		wS = np.array([cloud.tipicality for cloud in self.c])/tS
		for w,cloud in zip(wS,self.c):
			if w > wMax:
				wMax = w
				Best_cloud = cloud
		if self.win_all:
			Best_cloud.rnn.fit(y,z)
		if not self.win_all:
			for i,cloud in enumerate(self.c):
				cloud.rnn.fit(y,z)

	def predict(self,y):
		wSum = sum([cloud.tipicality for cloud in self.c])
		ws = np.array([cloud.tipicality/wSum for cloud in self.c]).reshape(-1,1)
		p = np.array([cloud.rnn.PredSingle(y) for cloud in self.c])
		p1 = (p*ws)
		p1 = sum([p1[i][-1] for i in range(len(p))])
		if self.win_all:
			p1 = p[np.argmax(ws)][-1]
		return p1
	
	def predict2(self,y):
		wSum = sum([cloud.tipicality for cloud in self.c])
		ws = np.array([cloud.tipicality/wSum for cloud in self.c]).reshape(-1,1)
		p = np.array([cloud.rnn.PredSingle(y) for cloud in self.c])
		p1 = (p*ws)
		p1 = sum([p1[i][-1] for i in range(len(p))])
		if self.win_all:
			p1 = p[np.argmax(ws)][-1]
		return p1
	
	def propagate(self,y, RNNs):
		wSum = sum([cloud.tipicality for cloud in self.c])
		ws = np.array([cloud.tipicality/wSum for cloud in self.c]).reshape(-1,1)
		p = np.array([rnn.PredSingle(y) for rnn in RNNs])
		p1 = (p*ws)
		p1 = sum([p1[i][-1] for i in range(len(p))])
		if self.win_all:
			p1 = p[np.argmax(ws)][-1]
		return p1
	
	def restore_rnn(self):
		for cloud in self.c:
			cloud.rnn.restore()

	def RUL_single(self,X,Scut=None,wta=False):
		if wta: self.win_all = True
		if Scut is None: Scut = 160
		pP,rulP = [0 for i in range(2)]
		xP = X.copy()
		pP = self.predict(xP)*self.fator
		eR = np.abs(self.HI[-1]-pP) 
		self.rls.update(np.abs(pP), eR)

		while xP[-1]>self.eol:
			pP = self.predict(xP)*self.fator
			xP = np.delete(np.append(xP,pP),0)
			rulP=rulP+1
			if rulP == Scut: 
				rulP = 0
				break

		self.restore_rnn()
		self.rulP = np.append(self.rulP,rulP)
		self.rulL = np.append(self.rulL,0)
		self.rulU = np.append(self.rulU,0)

		return
	
	def RulCalc(self,X,lower=True):
		x = X.copy()
		rul = 0
		RNNs_l,RNNs_u,RNNs_L,RNNs_U = [np.array([copy.deepcopy(cloud.rnn) for cloud in self.c]) for i in range(4)]
		while x[-1] > self.eol:
			p_l = self.propagate(x,RNNs_l)*self.fator
			p_u = self.propagate(x,RNNs_u)*self.fator
			P_L = self.propagate(x,RNNs_L)*self.fator
			P_U = self.propagate(x,RNNs_U)*self.fator
			e_l = abs(self.rls.predict(np.abs(p_l)))*self.ep
			e_u = abs(self.rls.predict(np.abs(p_u)))*self.ep
			e_L = abs(self.rls.predict(np.abs(P_L)))*self.ep
			e_U = abs(self.rls.predict(np.abs(P_U)))*self.ep

			pS = ([p_l-e_l, p_l+e_l, p_u-e_u, p_u+e_u, P_L-e_L, P_L+e_U, P_L-e_U, P_U+e_U])
			pMin = np.min(pS)
			pMax = np.max(pS)
			if lower: x = np.delete(np.append(x,pMin),0)
			elif not lower: x = np.delete(np.append(x,pMax),0)
			rul=rul+1
			if rul == self.lim: break
		return rul
	
	def RUL_uncertainty(self,X,Y):
		xP, rulP = X.copy(), 0
		yP = self.predict(xP)*self.fator
		yR = Y[-1]
		εR = abs(yP-yR)
		
		self.rls.update(np.abs(yP), εR)
		RNNs = np.array([copy.deepcopy(cloud.rnn) for cloud in self.c])
		while xP[-1]>self.eol:
			yP = self.propagate(xP,RNNs)*self.fator
			xP = np.delete(np.append(xP,yP),0)
			rulP=rulP+1
			if rulP == self.lim: break
		
		self.rulL = np.append(self.rulL,self.RulCalc(X,lower=True))
		self.rulU = np.append(self.rulU,self.RulCalc(X,lower=False))
		self.rulP = np.append(self.rulP,rulP)

		return
	
	def AddRUL(self):
		for cloud in self.cloud_activation:
			cloud.rul = np.append(cloud.rul,self.rulP[-1])
		self.cloud_activation = []
	
	def mergeClouds(self,x):
		
		i=0
		while(i<len(self.listIntersection)-1):
			merge=False
			j=i+1
			while(j<len(self.listIntersection)):
				if(self.listIntersection[i] == 1 and self.listIntersection[j] == 1):
					self.matrixIntersection[i,j] = self.matrixIntersection[i,j] + 1
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
				rI = self.c[i].R
				rJ = self.c[j].R
				spI = self.c[i].sp
				spJ = self.c[j].sp
				spSumI = self.c[i].spSum
				spSumJ = self.c[j].spSum
				covI = self.c[i].cov
				covJ = self.c[j].cov
				dIJ = euclidian_dist(x1=meanI, x2=meanJ)
				nIntersc = self.matrixIntersection[i,j]

				if (nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
	
					merge = True
					self.gCreated = self.gCreated + 1
					n = nI + nJ - nIntersc
					mean = ((nI * meanI) + (nJ * meanJ))/(nI + nJ)
					meant = ((nI * meantI) + (nJ * meantJ))/(nI + nJ)
					variance = ((nI - 1) * varianceI + (nJ - 1) * varianceJ)/(nI + nJ - 2)
					tipicality = ((nI*tipicalityI)+(nJ*tipicalityJ))/(nI + nJ)
					sp = ((nI*spI)+(nJ*spJ))/(nI + nJ)
					#spSum = ((nI*spSumI)+(nJ*spSumJ))/(nI + nJ)
					spSum = ((nI*tipicalityI)+(nJ*tipicalityJ))/(tipicalityI + tipicalityJ)
					cov = n/self.k

					R = (rI+rJ+dIJ)/2

					wIN = ((winI*tipicalityI)+(winJ*tipicalityJ))/(tipicalityI+tipicalityJ)
					wHS = ((wrecI*tipicalityI)+(wrecJ*tipicalityJ))/(tipicalityI+tipicalityJ)
					wOUT = ((woutI*tipicalityI)+(woutJ*tipicalityJ))/(tipicalityI+tipicalityJ)
					hI = ((hiI*tipicalityI)+(hiJ*tipicalityJ))/(tipicalityI+tipicalityJ)

					newCloud = DataCloud(x=x,ID=self.gCreated,nI=self.nI,nR=self.nR,nO=self.nO,
									ηS=[self.N1,self.N2,self.N3],tau=self.tau,decay=self.decay)
					newCloud.m = self.m

					for id in trackI:
						newCloud.track.append(id)
					for id in trackJ:
						newCloud.track.append(id)
						
					newCloud.updateDataCloud(n,mean,meant,variance,tipicality)
					newCloud.sp = sp
					newCloud.spSum = spSum
					newCloud.cov = cov
					newCloud.R = R
					newCloud.Rvec[0] = newCloud.R

					x_ = self.c[i].x + self.c[j].x
					t = self.c[i].t + self.c[j].t
					mat = np.array(list(zip(t,x_)), dtype=object)
					col = mat[:, 0].astype(int) 
					_, index = np.unique(col, return_index=True)
					result = mat[np.sort(index)]
					t = result[:, 0].tolist()
					x_ = result[:, 1].tolist()
					newCloud.x = x_
					newCloud.t = t
					
					
				
					newCloud.rnn.wIN = wIN
					newCloud.rnn.wHS = wHS
					newCloud.rnn.wOUT = wOUT
					newCloud.rnn.hI = hI
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
			if(merge): i = 0
			else: i += 1

	def run(self,X):
		self.aux = np.array([])
		self.aux2 = np.array([])
		if self.k == 1: self.xI = X
		self.listIntersection = np.zeros((np.size(self.c)),dtype=int)

		if self.k==1:
			self.CreateCloud(X)
			self.c[0].t.append(self.k)
			self.aux = np.append(self.aux,1)
			self.aux2 = np.append(self.aux,self.c[0].track)
			self.cloud_activation.append(self.c[0])

		elif self.k==2:
			self.c[0].addDataClaud(X)
			self.c[0].x.append(X)
			self.c[0].t.append(self.k)
			self.aux = np.append(self.aux,1)
			self.aux2 = np.append(self.aux,self.c[0].track)
			self.cloud_activation.append(self.c[0])

		elif self.k>=3:
			i=0
			createCloud = True
			self.alfa = np.zeros((np.size(self.c)),dtype=float)
			for cloud in self.c:
				n= cloud.n +1
				mean = ((n-1)/n)*cloud.mean + (1/n)*X
				meant = ((n-1)/n) * cloud.meant + (X.dot(X)/n)
				variance=meant-mean.dot(mean)
				eccentricity = (1/n)+((mean-X).T.dot(mean-X))/(n*variance)
				typicality = 1 - (eccentricity-(1e-12))
				norm_eccentricity = eccentricity/2
				norm_typicality = typicality/(self.k-2)
				cloud.eccAn = eccentricity
				if(norm_eccentricity<=(self.m**2 +1)/(2*n)):
					createCloud= False
					cloud.updateDataCloud(n,mean,meant,variance,typicality)
					self.alfa[i] = norm_typicality
					self.listIntersection.itemset(i,1)
					cloud.x.append(X)
					cloud.t.append(self.k)
					cloud.xF = X
					self.aux = np.append(self.aux,cloud.ID)
					self.aux2 = np.append(self.aux2,cloud.track)
					self.cloud_activation.append(cloud)
				else:
					self.alfa[i] = 0
					self.listIntersection.itemset(i,0)
				i+=1

			if(createCloud):
				self.CreateCloud(X)
				self.listIntersection = np.insert(self.listIntersection,i,1)
				self.matrixIntersection = np.pad(self.matrixIntersection, ((0,1),(0,1)), 'constant', constant_values=(0)) 
				self.c[-1].t.append(self.k)
				self.aux = np.append(self.aux,self.c[-1].ID)
				self.aux2 = np.append(self.aux2,self.c[-1].track)
				self.cloud_activation.append(self.c[-1])
			self.mergeClouds(X)

		for cloud in self.c:
			cloud.k = self.k

		self.cycleP = np.append(self.cycleP,self.st+self.k-1)
		self.rulR = np.flip(self.cycleP-(self.st-1))
		self.rulR_Train = np.append(self.rulR_Train,self.end-self.k)
		self.k=self.k+1

	def reset_rul(self):
		self.eolX = 0
		self.HI  = np.array([1])
		self.cycleP = np.array([])
		self.rulR = np.array([])
		self.rulL = np.array([])
		self.rulU = np.array([])
		self.cloud_activation = []
		self.rulP = np.array([])
		#self.OffLine_granules = None
		#self.cycleP = np.append(self.cycleP,self.st)
		#for cloud in self.c:
		#	cloud.rnn.restore_ni2()