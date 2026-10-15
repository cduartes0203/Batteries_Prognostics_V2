import numpy as np

class RLS_ExponentialRegressor:
    def __init__(self, lambda_=0.99, delta=1000):
        """
        Regressão exponencial recursiva:
        y = beta_0 * exp(beta_1 * x)

        Após log-transformação: ln(y) = ln(beta_0) + beta_1 * x
        """
        self.theta = np.zeros((2, 1))  # [theta_0 = ln(beta_0), theta_1 = beta_1]
        self.P = delta * np.eye(2)
        self.lambda_ = lambda_

    def _phi(self, x):
        return np.array([[1, x]]).T

    def update(self, x, y):
        if y <= 0:
            raise ValueError("y deve ser positivo para aplicar log em regressão exponencial.")
        phi_x = self._phi(x)
        y_trans = np.log(y)
        K = self.P @ phi_x / (self.lambda_ + phi_x.T @ self.P @ phi_x)
        error = y_trans - phi_x.T @ self.theta
        self.theta += K * error
        self.P = (self.P - K @ phi_x.T @ self.P) / self.lambda_

    def predict(self, x):
        phi_x = self._phi(x)
        ln_y = float(phi_x.T @ self.theta)
        #print('ln_y:',ln_y)
        return np.exp(ln_y)

    def get_params(self):
        """Retorna beta_0 e beta_1"""
        theta_0, theta_1 = self.theta.flatten()
        beta_0 = np.exp(theta_0)
        beta_1 = theta_1
        return beta_0, beta_1
    
class RLS_LogarithmicRegressor:
    def __init__(self, lambda_=0.9, delta=10000):
        """
        Regressão logarítmica recursiva via RLS.
        y = beta_0 + beta_1 * ln(x)

        :param lambda_: Fator de esquecimento
        :param delta: Valor inicial para P (confiança inicial baixa)
        """
        self.delta = delta
        self.theta = np.zeros((3, 1))              # [beta_0, beta_1]
        self.P = self.delta * np.eye(3)
        self.lambda_ = lambda_

    def _phi(self, x):
        """Transforma x em vetor de regressão [1, ln(x)]"""
        if x <= 0:
            raise ValueError("x deve ser positivo para regressão logarítmica.")
        return np.array([[1, np.log(x), np.log(x)**2]]).T

    def update(self, x, y):
        phi_x = self._phi(x)
        #print(phi_x)
        K = self.P @ phi_x / (self.lambda_ + phi_x.T @ self.P @ phi_x)
        error = y - phi_x.T @ self.theta
        self.theta += K * error
        self.P = (self.P - K @ phi_x.T @ self.P) / self.lambda_

    def predict(self, x):
        phi_x = self._phi(x)
        #return float(phi_x.T @ self.theta)
        return ((phi_x.T @ self.theta)).item()

    def get_params(self):
        return self.theta.flatten()
    
class RLS_VDF():
    def __init__(self,ns,nf,ff,df,dt):
        nf=nf+1
        self.parameters = nf
        self.w_0 = np.zeros(nf)
        self.w =  self.w_0
        self.lamda = ff
        self.e = df
        self.P = dt*np.eye(self.parameters)
        self.A = np.linalg.inv(np.eye(self.parameters,self.parameters))
        self.n_samples = ns
        
    def adapt(self,phi,y,wS=1):
        #phi = self.Afine(phi)
        #print('phi:',phi)
        for i in range(self.n_samples-1):
            print('-------------------------------------------')
            U, Sigma, VT = np.linalg.svd(self.P)
            psi_k = phi @ U
            column_norms = np.linalg.norm(psi_k, axis=0)
            lamda_bar = np.sqrt(self.lamda) * np.eye(self.parameters,self.parameters)
            for i in range(len(column_norms)):
                if column_norms[i] <= self.e: 
                    lamda_bar[i, i] = 1 
            lamda_k = U@lamda_bar@U.T   
            pbar_k = np.linalg.inv(lamda_k)@self.P@np.linalg.inv(lamda_k)
            print('wS:',wS)
            print('phi:',phi)
            print('pbar_k:',pbar_k)
            print('prod:',np.linalg.inv(np.eye(phi.shape[0]) + wS*phi @ pbar_k @ phi.T))
            self.P = (pbar_k - pbar_k@(phi.T)@(np.linalg.inv(np.eye(phi.shape[0]) + wS*phi @ pbar_k @ phi.T))@phi@pbar_k)
            self.w = self.w + wS*self.P@(phi.T)@(y - phi@self.w) + self.P@((lamda_k@self.A@lamda_k - self.A)@(self.w_0 - self.w))
            self.A = lamda_k@self.A@lamda_k + phi.T@phi

        return
    
    def predict(self,X):
        #X = np.append(1,X)
        #print('X:',X)
        return self.w@X
    
    def Afine(self,m):
        L,C = m.shape
        col = np.ones(L).reshape(-1,1)
        m = np.hstack((col, m))
        return m
    
