import numpy as np

class RLS():
    def __init__(self, nf, ff=1.0, dt=1000.0, xs=1.0):
        """
        Regressão polinomial recursiva (RLS) em x escalar, com termo afim:
        y = w_0 + w_1*(x/xs) + w_2*(x/xs)^2 + ... + w_nf*(x/xs)^nf

        :param nf: grau do polinômio (nf+1 parâmetros, incluindo o termo afim w_0)
        :param ff: fator de esquecimento (0 < ff <= 1)
        :param dt: valor inicial da diagonal de P (confiança inicial baixa)
        :param xs: escala de x (x/xs ~ O(1) melhora o condicionamento de P para graus altos)
        """
        if int(nf) != nf or nf < 0:
            raise ValueError("nf deve ser um inteiro >= 0 (grau do polinômio).")
        self.nf = int(nf)
        self.parameters = self.nf + 1
        self.w = np.zeros(self.parameters)
        self.lamda = ff
        self.xs = xs
        self.P = dt * np.eye(self.parameters)

    def _phi(self, x):
        """Vetor de regressão [1, x, x^2, ..., x^nf] (x escalonado por xs)"""
        return (float(x) / self.xs) ** np.arange(self.parameters)

    def update(self, x, y):
        phi = self._phi(x)
        e = y - self.w @ phi
        P_phi = self.P @ phi
        denominator = self.lamda + phi @ P_phi
        k = P_phi / denominator
        self.P = (1 / self.lamda) * (self.P - np.outer(k, P_phi))
        self.P = 0.5 * (self.P + self.P.T)          # mantém P simétrica numericamente
        self.w = self.w + k * e

    def predict(self, x):
        return float(self.w @ self._phi(x))

    def get_params(self):
        """Retorna [w_0 (termo afim), w_1, ..., w_nf]"""
        return self.w.copy()

class RLS_ExponentialRegressor:
    def __init__(self, ff=0.99, delta=1000):
        """
        Regressão exponencial recursiva:
        y = beta_0 * exp(beta_1 * x)

        Após log-transformação: ln(y) = ln(beta_0) + beta_1 * x
        """
        self.theta = np.zeros((2, 1))  # [theta_0 = ln(beta_0), theta_1 = beta_1]
        self.P = delta * np.eye(2)
        self.ff = ff

    def _phi(self, x):
        return np.array([[1, x]]).T

    def update(self, x, y):
        if y <= 0:
            raise ValueError("y deve ser positivo para aplicar log em regressão exponencial.")
        phi_x = self._phi(x)
        y_trans = np.log(y)
        K = self.P @ phi_x / (self.ff + phi_x.T @ self.P @ phi_x)
        error = y_trans - phi_x.T @ self.theta
        self.theta += K * error
        self.P = (self.P - K @ phi_x.T @ self.P) / self.ff

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
    def __init__(self, ff=0.9, dt=10000):
        """
        Regressão logarítmica recursiva via RLS.
        y = beta_0 + beta_1 * ln(x)

        :param ff: Fator de esquecimento
        :param delta: Valor inicial para P (confiança inicial baixa)
        """
        self.dt = dt
        self.theta = np.zeros((3, 1))              # [beta_0, beta_1]
        self.P = self.dt * np.eye(3)
        self.ff = ff

    def _phi(self, x):
        """Transforma x em vetor de regressão [1, ln(x)]"""
        if x <= 0:
            raise ValueError("x deve ser positivo para regressão logarítmica.")
        return np.array([[1, np.log(x), np.log(x)**2]]).T

    def update(self, x, y):
        phi_x = self._phi(x)
        #print(phi_x)
        K = self.P @ phi_x / (self.ff + phi_x.T @ self.P @ phi_x)
        error = y - phi_x.T @ self.theta
        self.theta += K * error
        self.P = (self.P - K @ phi_x.T @ self.P) / self.ff

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
        self.A = np.linalg.inv(self.P)     # matriz de informação: A = P^-1 (apenas para monitoramento)
        self.n_samples = ns                # mantido por compatibilidade; adapt faz uma única atualização por chamada

    def adapt(self,phi,y,wS=1):
        # Uma atualização por chamada: phi (p x nf) pode ser uma linha ou um lote; y (p,)
        # Minimiza J_{k+1}(w) = J_k(w) esquecido via Lambda_k + wS*||y - phi w||^2 (prior esquecido junto com os dados)
        #phi = self.Afine(phi)
        phi = np.atleast_2d(phi)
        U, Sigma, VT = np.linalg.svd(self.P)
        psi_k = phi @ U
        column_norms = np.linalg.norm(psi_k, axis=0)
        # esquece (sqrt(lambda)) só nas direções excitadas; nas demais, 1
        lamda_bar = np.where(column_norms > self.e, np.sqrt(self.lamda), 1.0)
        lamda_k = U@np.diag(lamda_bar)@U.T
        lamda_k_inv = U@np.diag(1/lamda_bar)@U.T
        pbar_k = lamda_k_inv@self.P@lamda_k_inv
        # (pbar_k^-1 + wS*phi'phi)^-1 pelo lema de inversão de matrizes
        self.P = pbar_k - wS*pbar_k@(phi.T)@np.linalg.inv(np.eye(phi.shape[0]) + wS*phi@pbar_k@phi.T)@phi@pbar_k
        self.w = self.w + wS*self.P@(phi.T)@(y - phi@self.w)
        self.A = lamda_k@self.A@lamda_k + wS*phi.T@phi

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
    
