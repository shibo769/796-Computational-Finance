# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:04:05 2021

@author: Shi Bo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

class CEV:
    def __init__(self, S0, K, r, T, trials, steps, sigma, beta):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.trials = trials
        self.steps = steps
        self.sigma = sigma
        self.beta = beta
        self.d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / \
            (self.sigma * np.sqrt(self.T))
        self.d2 = (np.log(self.S0 / self.K) + (self.r - self.sigma ** 2 / 2) * self.T) / \
            (self.sigma * np.sqrt(self.T))
    
    def simulation(self):
        dt = self.T / self.steps
        result = list()
        for i in range(self.trials):
            ds = self.S0
            for j in range(self.steps):
                dW = np.random.normal(0, np.sqrt(dt), size = self.T * self.steps)
                ds = ds + self.r * ds * dt + self.sigma * (ds ** self.beta) * dW[j]
            result.append(ds)
        return result
    
    def simu_price(self,df):
        price = np.mean(df)
        return price
    
    def simu_payoff(self,df):
        Payoff = list()
        for i in df:
            if i > self.K:
                Payoff.append(i - self.K)
            else: 
                Payoff.append(0)
        return np.mean(Payoff) * np.exp(-self.r * self.T)
    
    def BSM(self):
        Nd1 = stats.norm.cdf(self.d1)
        Nd2 = stats.norm.cdf(self.d2)
        price = self.S0 * Nd1 - Nd2 * self.K * np.exp(-self.r * self.T)
        return price
    
    def delta(self):
        self.delta = stats.norm.cdf((np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) \
                                * self.T) / (self.sigma * np.sqrt(self.T)))
        return self.delta
    
    def delta_neutral_payoff(self,df):
        result = list()
        for i in df:
            if i > self.K:
                result.append(i - self.K + self.delta * (self.S0 - i))
            else:
                result.append(self.delta * (self.S0 - i))
        return np.mean(result) * np.exp(-self.r * self.T)
        

if __name__ == '__main__':
    ################################(b)################################
    Eucall = CEV(100,100,0.0,1,1000,252,0.25,1.0)
    simu = Eucall.simulation()
    simu_price = Eucall.simu_price(simu)
    payoff = Eucall.simu_payoff(simu)
    print('(b)The simulation price of Euro call with CEV model is: %.6f' % payoff + '.')
    ################################(c)################################
    BSM_P = Eucall.BSM()
    print('(c)The price via BSM formula is: %.4f' % BSM_P + '.')
    ################################(d)################################
    Delta = Eucall.delta()
    print('(d)The delta of an at-the-money European call option with one year to expiry\
    is : {}'.format(Delta))
    ################################(e)################################
    print('(e)The number of stock needed to hedge one unit delta is {}.'.format(Delta))
    ################################(f)################################
    delta_neu = Eucall.delta_neutral_payoff(simu)
    print('(f)Delta neutral price with simulation is {}'.format(delta_neu))
    ################################(g)################################
    Eucall_b = CEV(100,100,0.0,1,10000,252,0.25,0.5)
    simu_diffB = Eucall_b.simulation()
    payoff_b = Eucall_b.simu_payoff(simu_diffB)
    print('(g)The portfolio payoff with 0.5 beta is %.4f' % payoff_b)
    ################################(h)################################
    Eucall_s = CEV(100,100,0.0,1,10000,252,0.4,1.0)
    simu_diffS = Eucall_s.simulation()
    payoff_s = Eucall_s.simu_payoff(simu_diffS)
    print('(h)The portfolio payoff with 0.4 sigma is %.4f' % payoff_s)