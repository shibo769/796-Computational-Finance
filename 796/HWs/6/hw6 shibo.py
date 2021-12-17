import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import root

class Euro_option():
    def __init__(self, K, S0, r, sigma, T):
        self.K = K
        self.S0 = S0
        self.r = r
        self.sigma = sigma 
        self.T = T
        
    def BSM(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2)
              * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        N_d1 = stats.norm.cdf(d1)
        N_d2 = stats.norm.cdf(d2)

        C = self.S0 * N_d1 - self.K * np.exp(-self.r * self.T) * N_d2
        P = C + self.K * np.exp(-self.r * self.T) - self.S0

        price = {'Put': P, 'Call': C}

        self.d1 = d1
        self.delta = N_d1

        return price
    
class Numerical_PDE():
    def __init__(self, S0, K1, K2, r, T, sigma, smin, smax, M, N):
        self.S0 = S0
        self.K1 = K1
        self.K2 = K2
        self.r = r
        self.T = T
        self.sigma = sigma
        self.smin = smin
        self.smax = smax
        self.M = M
        self.N = N
        
        self.ht = T / N
        self.hs = smax / M
        
        self.si = np.arange(self.smin, self.smax + self.hs, self.hs)
        
    def A(self):
        sigma = self.sigma
        ht = self.ht
        hs = self.hs
        r = self.r
        si = self.si
        M = self.M
        
        self.ai = (1 - (sigma ** 2 * si ** 2) * (ht / hs ** 2) - r * ht)
        self.li = (sigma ** 2 * si ** 2 / 2) * (ht / hs ** 2) - r * si * ht / (2 * hs)
        self.ui = (sigma ** 2 * si ** 2 / 2) * (ht / hs ** 2) + r * si * ht / (2 * hs)
        
        self.A = np.diag(self.ai[1:M])
        np.fill_diagonal(self.A[1:, :-1], self.li[2:M])
        np.fill_diagonal(self.A[:-1, 1:], self.ui[1:M - 1])
        
        return self.A
        
    def eigenvalue(self):
        eng_value, eng_vector = np.linalg.eig(self.A())
        eng_value = abs(eng_value)
        
        for i in eng_value:
            if abs(i) > 1:
                print('Absolute eigenvalue below 1',i)
        
        plt.figure()
        plt.plot(eng_value)
        plt.grid(True)
        plt.ylabel("Eigenvalue")
        plt.xlabel('M')
        plt.title("Eigenvalue of matrix A", fontsize = 15)
        plt.hlines(y = 1, xmin = 0, xmax = 300, linestyle = 'dashed',color = 'red')
        plt.show()
        
        return eng_value
    
    def call_price(self, Type = 'European'):
        si = self.si
        K1 = self.K1
        K2 = self.K2
        ht = self.ht
        hs = self.hs
        sigma = self.sigma
        r = self.r
        A = self.A()
        self.ui = (sigma ** 2 * si ** 2 / 2) * (ht / hs ** 2) + r * si * ht / (2 * hs)
        
        call1 = np.maximum(si - K1, 0)
        call2 = np.maximum(si - K2, 0)
        call = call1 - call2
        callt = call[1:M]
        
        for j in range(N):
            bj = self.ui[-1] * (K2 - K1) * np.exp(-self.r * j * self.ht)
            callt = A.dot(callt)
            callt[-1] = callt[-1] + bj
            if Type == 'American':
                callt = [max(x, y) for x, y in zip(callt, call[1:M])]
        
        return np.interp(self.S0, si[1:M], callt)
            
if __name__ == '__main__':
    S0 = 381.42
    K1 = 385
    K2 = 390
    r = 0.04
    K =385
    T = 7/12
    
    smax = 600
    smin = 0
    N = 5000
    M = 300
#################(3)#################
    sigma1 = 0.219
    sigma2 = 0.1909
    sigma = (sigma1 + sigma2) / 2
    print('The Volatility is %.4f' % sigma)
#################(5)#################
    PDE = Numerical_PDE(S0, K1, K2, r, T, sigma, smin, smax, M, N)
    PDE.eigenvalue()
#################(6)#################
    print('The call spread price without early execerise is: ',\
          Numerical_PDE(S0, K1, K2, r, T, sigma, smin, smax, M, N).call_price('European'))
#################(7)#################
    print('The call spread price with early execerise is: ',\
          Numerical_PDE(S0, K1, K2, r, T, sigma, smin, smax, M, N).call_price('American'))
#################(8)#################
    print('The early execerise premium is: ',
          abs(Numerical_PDE(S0, K1, K2, r, T, sigma, smin, smax, M, N).call_price('European') - 
          Numerical_PDE(S0, K1, K2, r, T, sigma, smin, smax, M, N).call_price('American')))