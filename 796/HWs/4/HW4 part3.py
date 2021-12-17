from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
import cmath
import warnings
from scipy.optimize import root

class FFT:
    def __init__(self, params, T = 3/12, S0 = 267.15, q = 0.0177, r = 0.015):
        self.kappa = params[0]
        self.theta = params[1]
        self.sigma = params[2]
        self.rho = params[3]
        self.v0 = params[4]
        self.T = T
        self.S0 = S0
        self.q = q
        self.r = r
        self.ii = complex(0, 1)
        
    def Heston_char(self, u):
        l = np.sqrt(self.sigma ** 2 * (u ** 2 + self.ii * u) + (self.kappa - self.ii * self.rho * self.sigma * u) ** 2)
        numer = np.exp(
            self.ii * u * np.log(self.S0) + self.ii * u * (self.r - self.q) * self.T + self.kappa * self.theta * self.T * (self.kappa - self.ii * self.rho * self.sigma * u) / self.sigma ** 2)
        denom = (np.cosh(l * self.T / 2) + (self.kappa - self.ii * self.rho * self.sigma * u) / l * np.sinh(l * self.T / 2)) ** (
                2 * self.kappa * self.theta / self.sigma ** 2)
        f = numer / denom * np.exp(-(u ** 2 + self.ii * u) * self.v0 / (l / np.tanh(l * self.T / 2) + self.kappa - self.ii * self.rho * self.sigma * u))
        return f
    
    def Heston_FFT(self, n, B, alpha, K):
        N = 2 ** n
        dv = B / N
        dk = 2 * np.pi / N / dv
        beta = np.log(self.S0) - dk * N / 2
        ii = self.ii
        
        vj = np.arange(0, N, dtype=complex) * dv
        km = beta + np.arange(0, N) * dk  
        
        delta_j_1 = np.zeros(N)
        delta_j_1[0] = 1

        Psi_vj = np.zeros(N, dtype=complex)

        for j in range(0, N):
            u = vj[j] - (alpha + 1) * ii
            numer = np.exp(-ii * beta * vj[j]) * self.Heston_char(u)
            denom = 2 * (alpha + vj[j] * ii) * (alpha + 1 + vj[j] * ii)
            Psi_vj[j] = numer / denom

        x = (2 - delta_j_1) * dv * Psi_vj
        z = np.fft.fft(x)

        Mul = np.exp(-alpha * np.array(km)) / np.pi
        Calls = np.exp(-self.r * self.T) * Mul * np.array(z).real

        K_list = list(np.exp(km))
        Call_list = list(Calls)
        tck = interpolate.splrep(K_list, Call_list)
        price = interpolate.splev(K, tck).real
        return price
    
def BSM(S0, r, K, T, sigma, q):
    d1 = (np.log(S0 / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
    d2 = d1 - sigma * T ** 0.5
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    price = S0 * Nd1 - Nd2 * K * np.exp(-r * T)
    return price, Nd1

def Heston_delta(params, h = 0.1, K = 275, T = 3/12, S0 = 267.15, q = 0.0177, r = 0.015):
    #平移S0 by h
    D1 = FFT(params, T, S0 + h, r, q).Heston_FFT(12, 1000, 1.5, K)
    D2 = FFT(params, T, S0 - h, r, q).Heston_FFT(12, 1000, 1.5, K)
    return (D1 - D2) / 2 / h

def Heston_vega(params, h = 0.01):
    Para1 = np.array(params) + np.array([0, 0.01, 0, 0, 0.01])
    Para2 = np.array(params) - np.array([0, 0.01, 0, 0, 0.01])
    V1 = FFT(Para1, T, S0 , r, q).Heston_FFT(12, 1000, 1.5, K)
    V2 = FFT(Para2, T, S0 , r, q).Heston_FFT(12, 1000, 1.5, K)
    return (V1 - V2) / 2 / h

def BSM_vega(sigma, K = 275, T = 3/12, S0 = 267.15, r = 0.015, q = 0.0177):
    d1 = (np.log(S0 / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
    vega = S0 * np.exp(-q * T) * np.sqrt(T) / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 / 2)
    return vega

if __name__ == '__main__':
     #From R part2
     K = 275
     T = 3/12
     S0 = 267.15
     q = 0.0177
     r = 0.015
     
     params = [3.52, 0.052, 1.18, -0.77, 0.034]
     print('The Heston delta is: ' , Heston_delta(params))

     sigma = root(lambda s: BSM(S0, r, K, T, s, q)[0] - FFT(params, 3 / 12).Heston_FFT(12, 1200, 1.5, 275),
         0.1).x[0]
     
     print('The BSM delta is:', BSM(S0, r, K, T, sigma, q)[1])
     
     print('The Heston vega is: ', Heston_vega(params))
     
     print('The BSM vega is: ', BSM_vega(sigma))