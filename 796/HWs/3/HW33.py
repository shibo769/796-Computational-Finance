import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import time
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from scipy.optimize import root


class FTT:
    def __init__(self, sigma, v0, k, rho, theta, S0, r, T, K):
        self.sigma = sigma
        self.v0 = v0
        self.k = k
        self.rho = rho
        self.theta = theta
        self.S0 = S0
        self.r = r
        self.T = T
        self.K = K
        self.ii = complex(0, 1)

    def dirac(self, n):
        result = np.zeros(len(n), dtype=complex)
        result[n == 0] = 1
        return result

    def char_func(self, u):
        l = np.sqrt(self.sigma ** 2 * (u ** 2 + self.ii * u) + (self.k - self.ii * self.rho * self.sigma * u) ** 2)
        numer = np.exp(
            self.ii * u * np.log(self.S0) + self.ii * u * (self.r - 0) * self.T + self.k * self.theta * self.T * (self.k - self.ii * self.rho * self.sigma * u) / self.sigma ** 2)
        denom = (np.cosh(l * self.T / 2) + (self.k - self.ii * self.rho * self.sigma * u) / l * np.sinh(l * self.T / 2)) ** (
                2 * self.k * self.theta / self.sigma ** 2)
        f = numer / denom * np.exp(-(u ** 2 + self.ii * u) * self.v0 / (l / np.tanh(l * self.T / 2) + self.k - self.ii * self.rho * self.sigma * u))
        return f
    
    def dirac(self, n):
        result = np.zeros(len(n), dtype=complex)
        result[n == 0] = 1
        return result
    
    def fft(self, alpha, n, B):
        t = time.time()
        
        ii = self.ii
        T = self.T
        r = self.r
        S0 = self.S0
        N = 2 ** n
        delta_v = B / N
        delta_k = 2 * np.pi / N / delta_v
        beta = np.log(S0) - delta_k * N / 2
        x = np.zeros(N, dtype=complex)
        km = np.zeros(N, dtype=complex)
        
        for j in range(1, N + 1):
            km[j - 1] = beta  + (j - 1) * delta_k
            vj = (j - 1) * delta_v
            fi = self.char_func(vj - (alpha + 1) * ii) / ((alpha + ii * vj) * (alpha + ii * vj + 1))
            x[j - 1] = np.exp(-ii * vj * beta) * fi
            
        J = np.arange(1, N + 1)
        w = (delta_v / 2) * (2 - self.dirac(J - 1))
        x = x * w
        y = np.fft.fft(x)
        K_list = np.exp(km)
        
        
        c = np.exp(-alpha * km) / np.pi * y.real
        list = interpolate.splrep(K_list, c)
        price = np.exp(-r * T) * interpolate.splev(self.K, list)
        runtime = time.time() - t
        return price, runtime

    def plot_alpha(self, alpha, n, B):
        y = np.array([self.fft(i, n, B)[0] for i in alpha])
        plt.figure(figsize=(10, 6), dpi=120)
        plt.plot(alpha, y)
        plt.title("European Call Option Price vs Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("European Call Option Price")
        plt.show()

    def plot_N_B(self, n_list, B_list, c):
        z = np.zeros((len(n_list), len(B_list)))
        e = np.zeros((len(n_list), len(B_list)))  # efficiency
        result = []
        x, y = np.meshgrid(n_list, B_list)
        for i in range(len(n_list)):
            for j in range(len(B_list)):
                temp = self.fft(1.1, n_list[i], B_list[j])
                z[i][j] = temp[0]
                e[i][j] = 1 / ((temp[0] - c) ** 2 * temp[1])
                result += [(e[i][j], n_list[i], B_list[j])]

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, z.T, rstride=1, cstride=1, cmap='coolwarm')
        plt.title("European Call Option Price vs N & B")
        ax.set_xlabel("n (N=2^n)")
        ax.set_ylabel("B")
        ax.set_zlabel("European Call Option Price")
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, e.T, rstride=1, cstride=1, cmap='coolwarm')
        plt.title("Efficiency vs N & B")
        ax.set_xlabel("n (N=2^n)")
        ax.set_ylabel("B")
        ax.set_zlabel("Efficiency")
        plt.show()

        return result

    def BSformula(self, K, T, sigma):
        S0 = self.S0
        r = self.r
        d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
        d2 = (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
        Nd1 = stats.norm.cdf(d1)
        Nd2 = stats.norm.cdf(d2)
        price = S0 * Nd1 - Nd2 * K * np.exp(-r * T)
        return price

    def K_vol(self, K_list, alpha, n, B):
        price_list = []
        K = self.K
        
        for i in K_list:
            self.K = i
            price_list += [self.fft(alpha, n, B)[0]]
            
        price_list = np.array(price_list)
        self.K = K

        vol = []
        
        for i in range(len(K_list)):
            vol += [root(lambda x: self.BSformula(K_list[i], self.T, x) - price_list[i], 0.3).x[0]]
            
        return price_list, vol
    
    def plot_K_price(self, K, price):
        plt.figure(figsize=(10, 6), dpi=120)
        plt.plot(K, price)
        plt.title("European Call Option Strike vs European Call Option Price")
        plt.xlabel("European Call Option Strike")
        plt.ylabel("European Call Option Price")
        plt.show()

    def T_vol(self, T_list, alpha, n, B):
        price_list = []
        T = self.T
        
        for i in T_list:
            self.T = i
            price_list += [self.fft(alpha, n, B)[0]]
            
        price_list = np.array(price_list)
        self.T = T

        vol = []
        
        for i in range(len(T_list)):
            vol += [root(lambda x: self.BSformula(self.K, T_list[i], x) - price_list[i], 0.3).x[0]]
            
        return price_list, vol

    def plot_T_price(self, T, price):
        plt.figure(figsize=(10, 6), dpi=120)
        plt.plot(T, price)
        plt.title("Time to Expiry vs European Call Option Price")
        plt.xlabel("Time to Expiry")
        plt.ylabel("European Call Option Price")
        plt.show()


if __name__ == '__main__':
#########################problem(a)(i)#########################
    alpha = 1.5
    sigma = 0.2
    v0 = 0.08
    k = 0.7
    rho = -0.4
    theta = 0.1
    S0 = 250
    K = 250
    r = 0.02
    T = 0.5
    n = 11
    B = 250 * 2.7
    a = FTT(sigma, v0, k, rho, theta, S0, r, T, K)
    alphas = np.linspace(-1, 38, 1000)
    a.plot_alpha(alphas, n, B)
#########################problem(a)(ii)########################
    B = np.linspace(250 * 2.5, 250 * 2.7, 100)
    n = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    result = a.plot_N_B(n, B, 21.27)
    print('The choosing parameters n is {} and B is {} to get the most efficiency'.format(max(result)[1], max(result)[2]))
#########################problem(a)(iii)#######################
    K2 = 260
    B = np.linspace(260 * 2.5, 260 * 2.7, 100)
    b = FTT(sigma, v0, k, rho, theta, S0, r, T, K2)
    result2 = b.plot_N_B(n, B, 16.73)
    print('The choosing parameters n is {} and B is {} to get the most efficiency'.format(max(result2)[1], max(result2)[2]))
#########################problem(b)(i)#########################
    alpha = 1.5
    sigma2 = 0.4
    v02 = 0.09
    k2 = 0.5
    rho2 = 0.25
    theta2 = 0.12
    S02 = 150
    K2 = 150
    r2 = 0.025
    T2 = 0.25
    c = FTT(sigma2, v02, k2, rho2, theta2, S02, r2, T2, K2)
    B = np.linspace(150 * 2.5, 150 * 2.7, 100)
    n = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    result3 = c.plot_N_B(n, B, 9.37)
    print('The most efficient n is {} and B is {}'.format(max(result3)[1], max(result3)[2]))
    n = 10
    B = 150 * 2.7
    alphas = np.linspace(0.1, 21, 100)
    c.plot_alpha(alphas, n, B)
    alpha = 1.5

    K_list = np.linspace(80, 230, 60)
    price_list, vol_list = c.K_vol(K_list, alpha, n, B)
    c.plot_K_price(K_list, price_list)
    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("European Call Option Strike vs Implied Volatility")
    plt.xlabel("European Call Option Strike")
    plt.ylabel("Implied Volatility")
    plt.plot(K_list, vol_list)
    plt.show()
#########################problem(b)(ii)########################
    T_list = np.linspace(1 / 12, 2, 100)
    price_list, vol_list = c.T_vol(T_list, alpha, n, B)
    c.plot_T_price(T_list, price_list)
    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("Time to Expiry vs Implied Volatility")
    plt.xlabel("Time to Expiry")
    plt.ylabel("Implied Volatility")
    plt.plot(T_list, vol_list)
    plt.show()
#########################problem(b)(iii)########################
    sigma = c.sigma
    sigma_list = [0.1, 0.3, 0.5, 0.7]
    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("European Call Option Strike vs Implied Volatility (change sigma)")
    plt.xlabel("European Call Option Strike")
    plt.ylabel("Implied Volatility")
    for i in sigma_list:
        c.sigma = i
        vol_list = c.K_vol(K_list, alpha, n, B)[1]
        plt.plot(K_list, vol_list)
    plt.legend(sigma_list)
    plt.show()

    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("Time to Expiry vs Implied Volatility (change sigma)")
    plt.xlabel("Time to Expiry")
    plt.ylabel("Implied Volatility")
    for i in sigma_list:
        c.sigma = i
        vol_list = c.T_vol(T_list, alpha, n, B)[1]
        plt.plot(T_list, vol_list)
    plt.legend(sigma_list)
    plt.show()
    c.sigma = sigma

    v0 = c.v0
    v0_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("European Call Option Strike vs Implied Volatility (change v0)")
    plt.xlabel("European Call Option Strike")
    plt.ylabel("Implied Volatility")
    for i in v0_list:
        c.v0 = i
        vol_list = c.K_vol(K_list, alpha, n, B)[1]
        plt.plot(K_list, vol_list)
    plt.legend(v0_list)
    plt.show()

    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("Time to Expiry vs Implied Volatility (change v0)")
    plt.xlabel("Time to Expiry")
    plt.ylabel("Implied Volatility")
    for i in v0_list:
        c.v0 = i
        vol_list = c.T_vol(T_list, alpha, n, B)[1]
        plt.plot(T_list, vol_list)
    plt.legend(v0_list)
    plt.show()
    c.v0 = v0

    k = c.k
    k_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("European Call Option Strike vs Implied Volatility (change k)")
    plt.xlabel("European Call Option Strike")
    plt.ylabel("Implied Volatility")
    for i in k_list:
        c.k = i
        vol_list = c.K_vol(K_list, alpha, n, B)[1]
        plt.plot(K_list, vol_list)
    plt.legend(k_list)
    plt.show()

    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("Time to Expiry vs Implied Volatility (change k)")
    plt.xlabel("Time to Expiry")
    plt.ylabel("Implied Volatility")
    for i in k_list:
        c.k = i
        vol_list = c.T_vol(T_list, alpha, n, B)[1]
        plt.plot(T_list, vol_list)
    plt.legend(k_list)
    plt.show()
    c.k = k

    rho = c.rho
    rho_list = [-0.4, -0.2, 0, 0.2, 0.4]
    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("European Call Option Strike vs Implied Volatility (change rho)")
    plt.xlabel("European Call Option Strike")
    plt.ylabel("Implied Volatility")
    for i in rho_list:
        c.rho = i
        vol_list = c.K_vol(K_list, alpha, n, B)[1]
        plt.plot(K_list, vol_list)
    plt.legend(rho_list)
    plt.show()

    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("Time to Expiry vs Implied Volatility (change rho)")
    plt.xlabel("Time to Expiry")
    plt.ylabel("Implied Volatility")
    for i in rho_list:
        c.rho = i
        vol_list = c.T_vol(T_list, alpha, n, B)[1]
        plt.plot(T_list, vol_list)
    plt.legend(rho_list)
    plt.show()
    c.rho = rho

    theta = c.theta
    theta_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("European Call Option Strike vs Implied Volatility (change theta)")
    plt.xlabel("European Call Option Strike")
    plt.ylabel("Implied Volatility")
    for i in theta_list:
        c.theta = i
        vol_list = c.K_vol(K_list, alpha, n, B)[1]
        plt.plot(K_list, vol_list)
    plt.legend(theta_list)
    plt.show()

    plt.figure(figsize=(10, 6), dpi=120)
    plt.title("Time to Expiry vs Implied Volatility (change theta)")
    plt.xlabel("Time to Expiry")
    plt.ylabel("Implied Volatility")
    for i in theta_list:
        c.theta = i
        vol_list = c.T_vol(T_list, alpha, n, B)[1]
        plt.plot(T_list, vol_list)
    plt.legend(theta_list)
    plt.show()
    c.theta = theta