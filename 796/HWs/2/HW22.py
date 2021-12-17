import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
np.set_printoptions(suppress=True)

class  Contingent():
    def __init__(self, r, S0, K, sigma, T):
        self.r = r
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * self.T ** 0.5)
        self.d2 = (np.log(self.S0 / self.K) + (self.r - self.sigma ** 2 / 2) * self.T) / (self.sigma * self.T ** 0.5)
        
    def d(self,x):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-np.square(x) / 2)

    def Gauss_d(self, x):
        mu = self.S0
        sd = self.sigma
        return 1 / (sd * (2 * np.pi) ** 0.5) * np.exp(-1 * (x - mu) ** 2 / (2 * sd ** 2))
    
    def Multi_d(self, x, y, sd, rho):
        mu = self.S0
        sd1 = self.sigma
        sd2 = sd
        e = np.exp(-1 / (2 * (1 - rho ** 2)) * ((x - mu) ** 2 / sd1 ** 2 + (y - mu) ** 2 / sd2 ** 2 
                                                - 2 * rho * (x - mu) * (y - mu) / (sd1 * sd2)))
        return (1 / (2 * np.pi * sd1 * sd2 * np.sqrt(1 - rho ** 2))) * e
    
    def Midpoint(self,N):
        mu = self.S0
        sd = self.sigma
        a = self.K
        b = mu + 5 * sd
        xi = np.array([a + (b - a) * (i + 0.5) / N for i in range(N)])
        integrand = np.exp(-self.r * self.T) * (xi - self.K) * self.Gauss_d(xi)
        temp_sum = np.sum(((b - a) / N * integrand))
        return temp_sum
        
    def contingent(self, sd2, K2, N, rho):
        mu = self.S0
        sd1 = self.sigma
        sd2 = sd2
        a1 = self.K
        b1 = mu + 5 * sd1
        a2 = mu - 5 * sd2
        b2 = K2
        x1i = np.array([a1 + (b1 - a1) / N * (i + 0.5) for i in range(N)])
        x2i = np.array([a2 + (b2 - a2) / N * (i + 0.5) for i in range(N)])
        temp_sum = 0
        for i in x2i:
            integrand = np.exp(-self.r * self.T) * (x1i - self.K) * self.Multi_d(x1i, i, sd2, rho)
            temp_sum += (b2 - a2) / N * (np.sum((b1 - a1) / N * integrand))
        return temp_sum
        
if __name__ == '__main__':
    ############################Problem 2 (1)############################
    S1 = yf.download('SPY', '2020-01-29')['Close'][0]
    S2 = yf.download('SPY', '2020-08-29')['Close'][0]
    K1 = 380
    K2 = 375
    t1 = 1
    t2 = 0.5
    r = 0
    sigma1 = 20
    sigma2 = 15
    rho = 0.95
    bound = 10
    
    Call = Contingent(r, S1, K1, sigma1, t1)
    vanilla = Call.Midpoint(100)
    print('The price of the one year call on SPY with the strike K1 = 380 is: ',vanilla)
    ############################Problem 2 (2)############################
    Con = Call.contingent(sigma2, K2, 100, rho)
    print('The contingent option price is: ', Con)
    ############################Problem 2 (3)############################
    rho_list = [0.8,0.5,0.2]
    for i in rho_list:
        CoN = Call.contingent(sigma2, K2, 100, i)
        print('The contingent option price with different rho are: ',CoN)
    ############################Problem 2 (5)############################
    SPY_list = [370,360]
    for i in SPY_list:
        CON = Call.contingent(sigma2, i, 100, rho)
        print('The contingent option price with different SPY 6 month are: ',CON)
        