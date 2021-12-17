import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import yfinance as yf
np.set_printoptions(suppress=True)

class Call:
    def __init__(self, r, S0, K, sigma, T, lower_bound):
        self.r = r
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * self.T ** 0.5)
        self.d2 = (np.log(self.S0 / self.K) + (self.r - self.sigma ** 2 / 2) * self.T) / (self.sigma * self.T ** 0.5)
        self.lower_bound = lower_bound
        
    def BSM(self):
        Nd1 = stats.norm.cdf(self.d1)
        Nd2 = stats.norm.cdf(self.d2)
        price = self.S0 * Nd1 - Nd2 * self.K * np.exp(-self.r * self.T)
        return price
    
    def density(self, x):
        return stats.norm.pdf(x)
    
    def reimann(self,Type, N, a, b):
        temp_sum = 0
        if Type == 'left_reimann':
            for i in range(N):
                xi = a + (b - a) * i / N 
                temp_sum += ((b - a) / N) * self.density(xi)
        elif Type == 'mid_reimann':
            for i in range(N):
                xi = a + (b - a) * (i - 1/2) / N
                temp_sum += ((b - a) / N) * self.density(xi)
        elif Type == 'Gauss_Node':
            x,w = np.polynomial.legendre.leggauss(N)
            y = (x * (b - a) + b + a) / 2
            temp_sum += sum(w * (self.density(y) * (b - a) / 2))
        return temp_sum
    
    def pricing(self, d11, d22):
        price = d11 * self.S0 - d22 * self.K * np.exp(-self.r * self.T)
        return price

    def Price(self, N, method):
        if method == 'left_riemann':
            price = self.pricing(self.reimann('left_reimann', N, self.lower_bound, self.d1),
                                  self.reimann('left_reimann', N, self.lower_bound, self.d2)) 
        elif method == 'mid_riemann':
            price = self.pricing(self.reimann('mid_reimann', N, self.lower_bound, self.d1),
                                  self.reimann('mid_reimann', N, self.lower_bound, self.d2)) 
        elif method == 'Gauss_Node':
            price = self.pricing(self.reimann('Gauss_Node', N, self.lower_bound, self.d1),
                                  self.reimann('Gauss_Node', N, self.lower_bound, self.d2)) 
        return price
    
    def error(self, df):
        error = abs(df - self.BSM())
        return error
    
    def plot(self, Nodes, methods):
        plt.figure(figsize = (10,6), dpi = 100)
        if methods == left_riemann:
            plt.plot(Nodes, call.error(methods))
            plt.plot(Nodes, 1 / Nodes ** 2)
            plt.plot(Nodes, 1 / Nodes ** 3)
            plt.title('Error of left riemann method', fontsize = 20)
            plt.legend(['Left riemann','O(N^-2)', 'O(N^-3)'])
        elif methods == mid_riemann:
            plt.plot(Nodes, call.error(methods))
            plt.plot(Nodes, 1 / Nodes ** 2)
            plt.plot(Nodes, 1 / Nodes ** 3)
            plt.title('Error of midpoint riemann method', fontsize = 20)
            plt.legend(['Midpoint riemann','O(N^-2)', 'O(N^-3)'])
        elif methods == Gauss_Node:
            plt.plot(Nodes, call.error(methods))
            plt.plot(Nodes, 1 / Nodes ** Nodes)
            plt.plot(Nodes, 1 / Nodes ** (2 * Nodes))
            plt.title('Error of gauss legendre method', fontsize = 20)
            plt.legend(['Gauss legendre','O(N^-N)', 'O(N^-2N)'])
        plt.xlabel('N',fontsize = 15)
        plt.ylabel('ABS error', fontsize = 15)
        plt.show()

if __name__ == "__main__":
    ############################Problem 1 (1)############################
    call = Call(0.01, 10, 12, 0.2, 1/4, -5)
    print('The price of the call calculated by BS formula is:', call.BSM(),'.\n')
    ############################Problem 1 (2)############################
    Nodes = [5,10,50]
    left_riemann = [call.Price(i,'left_riemann') for i in Nodes]
    mid_riemann = [call.Price(i,'mid_riemann') for i in Nodes]
    Gauss_Node = [call.Price(i,'Gauss_Node') for i in Nodes]
    Methods = [left_riemann,mid_riemann,Gauss_Node]
    error_left,error_mid,error_Gauss = [call.error(i) for i in Methods]
    print('left_riemann price: \n',left_riemann)
    print('mid_riemann price: \n',mid_riemann)
    print('Gauss_Node price: \n',Gauss_Node)
    print('Errors via left_riemann are: \n',error_left)
    print('Errors via mid_riemann are: \n',error_mid)
    print('Errors via Gauss_Node are: \n',error_Gauss)
    ############################Problem 1 (3)############################
    Nodes = np.array([i for i in range(1,50)])
    left_riemann = [call.Price(i,'left_riemann') for i in Nodes]
    mid_riemann = [call.Price(i,'mid_riemann') for i in Nodes]
    Gauss_Node = [call.Price(i,'Gauss_Node') for i in Nodes]

    methods = [left_riemann,mid_riemann,Gauss_Node]
    for i in methods:
        call.plot(Nodes,i)