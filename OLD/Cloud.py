import numpy as np
import math
from Functions.RLS import *
from OLD.RTLO import *

def euclidian_dist(x1,x2):
    return np.linalg.norm(x1 - x2,ord=None)


class DataCloud:
	N=0
	def __init__(self,x,ID,nI=1,nR=1,nO=1,ηS=[1,1,1],tau=1,decay=1):
		self.ID = ID
		self.track = [ID]
		self.merged = False
		self.merge = None
		self.m = None
		self.n=1
		self.k=1
		self.mean=x
		#self.meant=np.array(x).dot(np.array(x))
		#print(x,type(x))
		#self.meant=x.dot(x)
		#print(x.dot(x))
		self.meant=x.dot(x)
		self.variance=0
		self.pertinency=1
		self.tipicality=1e-12
		self.nS = len(x)
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
		self.x = [x]
		self.rul = np.array([])
		self.t = []
		self.R = 0
		self.Rsum = 0
		self.Rvec = np.array([0])
		self.Rmax = 0
		self.sp = 0
		self.spSum = 0
		self.cov = 0
		self.v = 0
		
	def addDataClaud(self,x):
		self.n=2
		self.mean=(self.mean+x)/2
		self.meant=((self.meant) + x.dot(x))/2
		self.variance=self.meant-self.mean.dot(self.mean)
		self.radius()

	def updateDataCloud(self,n,mean,meant,variance,tipicality):
		self.n=n
		self.mean=mean
		self.meant=meant
		self.variance=variance
		self.tipicality=tipicality
		self.radius()
		self.CalcSpecificity()

	def radius(self):
		v = self.variance
		n = self.n
		m = self.m
		R = np.sqrt(np.abs(v/n) * (((m**2)*(n+1))+1))
		self.Rsum = self.Rsum + R
		self.Rvec = np.append(self.Rvec,R)
		self.R = R
		
	def CalcSpecificity(self):
		n = len(self.mean)
		vR = ((math.pi**(n/2))/(math.gamma((n/2)+1)))*(self.R**n)
		vMax = ((math.pi**(n/2))/(math.gamma((n/2)+1)))*(1**n)
		local_sp = max(0,1-(vR/vMax))
		self.sp = local_sp
		self.spSum = self.spSum + local_sp
	
	def CalcV(self):
		self.CalcSpecificity()
		sp = self.spSum/self.k
		cov = self.n/self.k
		self.cov = cov
		self.v = cov*sp
	