import numpy as np
import math
from Testing.DRTLO_GD_R2 import *
class DataCloud:
	N=0
	def __init__(self,x,h,ID,rho,m,nI=1,nR=1,nO=1,ηS=[1,1,1],tau=1,mode='past',act = 'tanh'):
		self.ID = ID
		self.track = [ID]
		self.merged = False
		self.merge = None
		self.n = 1
		self.dim = len(x)
		self.rho = rho
		self.m = m
		self.mean=x
		self.center=x
		self.meant=np.array(x).dot(np.array(x))
		self.ecc = np.array([])
		self.variance=0
		self.pertinency=1
		self.tipicality=1e-12
		self.nI = nI
		self.nR = nR
		self.nO = nO
		self.N1 = ηS[0]
		self.N2 = ηS[1]
		self.N3 = ηS[2]
		self.mode = mode
		self.tau = tau
		self.act = act
		self.rnn = RTLO(self.nI, self.nR, self.nO,
					[self.N1, self.N2, self.N3], self.tau, self.mode, self.act)
		self.x = [x]
		self.rulP = np.array([])
		self.rulR = np.array([])
		self.rulM = []
		self.rulU = []
		self.rulL = []
		self.t = []
		self.t2 = []
		self.V_ = np.array([])
		self.R_ = np.array([])
		self.R = 0
		self.Rmax = 0
		self.Recc = 0
		self.Vecc = 0
		self.Vmax = 0
		self.xMax = x
		self.xMin = x
		self.xRef = x
		self.xIni = x
		self.sp = 0
		self.sp_hist = np.array([])
		self.sp_sum = 0
		self.cov = 1
		self.cov_hist = np.array([])
		self.V = 0
	
	def PrintParams(self):
		if self.ID != 'GM':
			print( f'Granule: G{self.ID} | Radius: {self.R:2.5f} | Volume {self.V:2.5f} | Coverage {self.cov:2.5f} | Specificity {self.sp:2.5f}')
		else:
			print( f'Granule: {self.ID} | Radius: {self.R:2.5f} | Volume {self.V:2.5f} | Coverage {self.cov:2.5f} | Specificity {self.sp:2.5f}')
		
		
	def AddDataClaud(self,x,t,store=False):
		self.n=2
		self.mean=(self.mean+x)/2
		self.meant=((self.meant)/2) + (x.dot(x))/2
		self.variance=self.meant-self.mean.dot(self.mean)
		if store:
			self.t.append(t)
			self.x.append(x)

	def UpdateDataCloud(self,n,mean,meant,variance,tipicality,t,x,store=False):
		self.n=n
		self.mean=mean
		self.meant=meant
		self.variance=variance
		self.tipicality=tipicality
		self.Recc = np.sqrt(self.variance * self.m**2)
		self.Vecc = (math.pi ** (self.dim / 2)) / math.gamma((self.dim / 2) + 1) * (self.Recc ** self.dim)

		if store:
			self.t.append(t)
			self.x.append(x)

	def AdjustBounds(self, x, k, density=False):
			dim = self.dim
			R = np.sqrt(self.variance*(self.m**2))
			self.R = R
			self.R_ = np.append(self.R_,R)
			self.V = (math.pi ** (dim / 2)) / math.gamma((dim / 2) + 1) * (self.R ** dim)

	