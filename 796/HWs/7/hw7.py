import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import root
from scipy import interpolate

class FFT(object):
    def __init__(self, params, T, S0, q, r):
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
    
class Simulation(object):
    def __init__(self, params, steps, N, T, S0, r, q):
        self.kappa = params[0]
        self.theta = params[1]
        self.sigma = params[2]
        self.rho = params[3]
        self.v0 = params[4]

        self.steps = steps
        self.N = N
        self.dt = T / steps

        self.T = T
        self.S0 = S0
        self.r = r
        self.q = q

    def sim_paths(self):
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        v0 = self.v0
        steps = self.steps
        N = self.N
        dt = self.dt
        S0 = self.S0
        r = self.r
        q = self.q
        T = self.T
        
         # Generate random Brownian Motion
        MU = np.array([0, 0])
        COV = np.matrix([[dt, rho * dt],[rho * dt, dt]])
        W = np.random.multivariate_normal(MU, COV, (N, steps))
        W_S = W[:, :, 0]
        W_V = W[:, :, 1]
        
        # Generate paths
        vt = np.zeros((N, steps))
        vt[:, 0] = v0
        St = np.zeros((N, steps))
        St[:, 0] = S0
        
        for t in range(1, steps):
            St[:, t] = St[:, t - 1] + (r - q) * St[:, t - 1] * dt + \
            np.sqrt(vt[:, t - 1].clip(min=0)) * St[:, t - 1] * W_S[:, t - 1]
            vt[:, t] = vt[:, t - 1] + kappa * (theta - vt[:, t - 1]) * dt + \
            sigma * np.sqrt(vt[:, t - 1].clip(min=0)) * W_V[:, t - 1]
            
        plt.plot(St)
        return St
    
    def euro_price(self, K):
        St = self.sim_paths()
        result = np.maximum(St[:,-1]-K, 0)

        return np.mean(result) * np.exp(-self.r * self.T)
    
    def up_and_out(self, K1, K2):
        St = self.sim_paths()
        max_S = np.max(St, axis = 1)
        indicator = np.where(max_S < K2, 1, 0)
        payoff = np.maximum(St[:, -1] - K1, 0) * indicator
        
        return np.exp(-self.r * self.T) * np.mean(payoff)
    
    def control_variate(self, K1, K2, expectation):
        St = self.sim_paths()
        max_S = np.max(St, axis = 1)
        indicator = np.where(max_S < K2, 1, 0)
        payoff_exotic = np.maximum(St[:, -1] - K1, 0) * indicator
        
        payoff_euro = np.maximum(St[:, -1] - K, 0)
        
        cov = np.cov(payoff_exotic, payoff_euro)
        c = -cov[0][1] / cov[0][0]
        
        payoff = payoff_exotic + c * (payoff_euro - expectation)
        return np.mean(payoff) * np.exp(-self.r * self.T)
    
if __name__ == '__main__':
##############################Problem 1##############################
    params = (3.52, 0.052, 1.18, -0.77, 0.034)
##############################Problem 3##############################
    S0 = 282
    r = 0.015
    q = 0.0177
    K = 285
    T = 1
    
    n = 11
    B = 1000
    alpha = 1.5
    FFT_price = FFT(params, T, S0, q, r).Heston_FFT(n, B, alpha, K)
    print('The price via FFT is: ', FFT_price)
    steps = 252
    
    Ns = (100, 500, 1000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 80000, 100000)
    Euro_price = []
    for i in Ns:
        Euro_price.append((Simulation(params, steps, i, T, S0, r, q).euro_price(K)))
        
    diff = abs(FFT_price - Euro_price)
    
    table1 = pd.DataFrame({'Ns' : Ns, 'Euro price' : Euro_price, 'Error' : diff})
    print('European option price for different N:\n',table1,'\n')
    
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Euro call price with different N', fontsize = 20)
    ax.set_xlabel('N', fontsize = 15)
    ax.set_ylabel('Option Price', fontsize = 15)
    ax.plot(Ns, Euro_price, 'o-')
    ##############################Problem 4##############################
    K1 = 285
    K2 = 315
    
    Ns = (100, 500, 1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 80000, 100000)
    up_out_price = []
    for i in Ns:
        up_out_price.append((Simulation(params, steps, i, T, S0, r, q).up_and_out(K1, K2)))
    
    table2 = pd.DataFrame({'Ns' : Ns, 'Up and out Price' : up_out_price})
    print('Up and out option price for different N: \n', table2, '\n')
    
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Up and out call with different N', fontsize = 20)
    ax.set_xlabel('N', fontsize = 15)
    ax.set_ylabel('Option Price', fontsize = 15)
    ax.plot(Ns, up_out_price, 'o-')
    ##############################Problem 5##############################
    expectation = Simulation(params, steps, 100000, T, S0, r, q).euro_price(K)
    
    Ns = (100, 500, 1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 80000, 100000)
    control_exotic_price = []
    for i in Ns:
        control_exotic_price.append(Simulation(params, steps, i, T, S0, r, q).control_variate(K1, K2, expectation))
    
    table3 = pd.DataFrame({'Ns' : Ns, 'Exotic price with control variate' : control_exotic_price})
    print('Up and out option Price with control variate:\n', table3)
    
    plt.figure(figsize = (10, 6))
    plt.plot(Ns, up_out_price, label = 'Exotic')
    plt.plot(Ns, control_exotic_price, label = 'Exotic with Control Variate')
    plt.title('Convergence Rate Comparison', fontsize = 20)
    plt.xlabel('N', fontsize = 15)
    plt.ylabel('Option Price', fontsize = 15)
    plt.legend(fontsize = 20)
    plt.show()