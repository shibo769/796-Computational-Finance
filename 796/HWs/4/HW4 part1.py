import math
import numpy as np
import pandas as pd
from scipy.stats import norm, kurtosis, skew, mode
from scipy.optimize import root, minimize, fsolve
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.stats as stats

def delta(K, sigma, T, S0, r, Type = 'call'):
    if Type not in ['call','put']:
        raise ValueError('Option type must be \'call\' or \'put\'')
    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    if Type == 'call':
        return stats.norm.cdf(d1)
    elif Type == 'put':
        return 1-stats.norm.cdf(d1)

def BSM(S0, r, K, T, sigma):
        d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
        d2 = (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
        Nd1 = stats.norm.cdf(d1)
        Nd2 = stats.norm.cdf(d2)
        price = S0 * Nd1 - Nd2 * K * np.exp(-r * T)
        return price
    
def risk_den(T):
    K_list = np.arange(80,110,0.01)
    vol = []
    h = 0.01
    if T == 1 / 12:
        vol = inter1M[1] + inter1M[0] * K_list
    else:
        vol = inter3M[1] + inter3M[0] * K_list
        
    call1 = BSM(S0, r, K_list - h, T, vol)
    call2 = BSM(S0, r, K_list, T, vol)
    call3 = BSM(S0, r, K_list + h, T, vol)
    
    result = (call1 - 2 * call2 + call3) / h ** 2
    
    return result, K_list, vol

def cons_risk_den(T):
    K_list = np.arange(60,140,0.01)
    vol = []
    h = 0.01
    if T == 1 / 12:
        vol = 0.1824
    else:
        vol = 0.1645
        
    call1 = BSM(S0, r, K_list - h, T, vol)
    call2 = BSM(S0, r, K_list, T, vol)
    call3 = BSM(S0, r, K_list + h, T, vol)
    
    result = (call1 - 2 * call2 + call3) / h ** 2
    
    return result, K_list

def digital(S_list, K, density_list, Type = 'call', h = 0.1):
    price = list()
    for i in range(1, len(S_list - 1)):
        if Type == 'call':
            if S_list[i] > K:
                price.append(density_list[i - 1] * h)
        if Type == 'put':
            if S_list[i] < K:
                price.append(density_list[i - 1] * h)
    return np.sum(price)
    
def call(density, S_list, K, Type, h):
    price = 0
    for i in range(1, len(S_list - 1)):
        if Type is "call":
            price += max(S_list[i] - K, 0) * density[i - 1] * h
        else:
            price += max(K - S_list[i], 0) * density[i - 1] * h
    return price


if __name__=='__main__':
#########################problem(1)(a)#########################
    S0 = 100
    r = 0
    T1 = 1 / 12
    T2 = 1 / 4
    vol_1M_put = [[0.1,0.3225],[0.25,0.2473],[0.4,0.2021],[0.5,0.1824]]
    vol_3M_put = [[0.1,0.2836],[0.25,0.2178],[0.4,0.1818],[0.5,0.1645]]
    vol_1M_call = [[0.1,0.1148],[0.25,0.1370],[0.4,0.1574]]
    vol_1M_call = reversed(vol_1M_call)
    vol_3M_call = [[0.1,0.1094],[0.25,0.1256],[0.4,0.1462]]
    vol_3M_call = reversed(vol_3M_call)
    
    K_1M = []
    K_3M = []
    
    for i in vol_1M_put:
        sol = fsolve(lambda k: delta(k,i[1],T1,S0,r,'put')-i[0], S0)
        K_1M.append(sol[0])
    for i in vol_3M_put:
        sol = fsolve(lambda k: delta(k,i[1],T2,S0,r,'put')-i[0], S0)
        K_3M.append(sol[0])
    for i in vol_1M_call:
        sol = fsolve(lambda k: delta(k,i[1],T1,S0,r,'call')-i[0], S0)
        K_1M.append(sol[0])
    for i in vol_3M_call:
        sol = fsolve(lambda k: delta(k,i[1],T2,S0,r,'call')-i[0], S0)
        K_3M.append(sol[0])
    
    K_dict = {'1M': K_1M, '3M': K_3M}
    strike = pd.DataFrame(K_dict,index=['10DP','25DP','40DP','50D','40DC','25DC','10DC'])
    print(strike)
    
#########################problem(1)(b)#########################
    vol_1M = [0.3225, 0.2473, 0.2021, 0.1824, 0.1574, 0.1370, 0.1148]
    vol_3M = [0.2836, 0.2178, 0.1818, 0.1645, 0.1462, 0.1256, 0.1094]
    
    inter1M = np.polyfit(K_1M, vol_1M,1)
    inter3M = np.polyfit(K_3M, vol_3M,1)
    print('\nThe interpolation fitted 1M coef and intercept are: ',inter1M)
    print('The interpolation fitted 3M coef and intercept are: ',inter3M)
    
#########################problem(1)(c)#########################
    den_1M = risk_den(1 / 12)
    den_3M = risk_den(1 / 4)
    
    plt.figure(dpi = 120)
    plt.plot(den_1M[1],den_1M[0], label = '1M')
    plt.plot(den_3M[1],den_3M[0], label = '3M')
    plt.title('Risk Neutral Density', fontsize = 15)
    plt.legend()
#########################problem(1)(d)#########################
    den_1M50 = cons_risk_den(1 / 12)
    den_3M50 = cons_risk_den(1 / 4)
    
    plt.figure(dpi = 120)
    plt.plot(den_1M50[1],den_1M50[0], label = '1M')
    plt.plot(den_3M50[1],den_3M50[0], label = '3M')
    plt.title('Risk Neutral Density with Constant Vol', fontsize = 15)
    plt.legend()
#########################problem(1)(e)#########################
    S_list = np.linspace(60,131,1000)
    dig_put = digital(S_list, 110, den_1M[0], Type = 'put', h = 0.1)
    print('The price of 1M European Digital Put Option with K = 110 is: ', dig_put)
    S_list = np.linspace(78,109,1000)
    dig_call = digital(S_list, 105, den_3M[0], Type = 'call', h = 0.1)
    print('The price of 3M European Digital call Option with K = 100 is: ', dig_call)
    
    new_vol_1M = den_1M[2]
    new_vol_3M = den_3M[2]
    vol_2M = [(new_vol_1M[i] + new_vol_3M[i]) / 2 for i in range(len(new_vol_1M))]
    
    K_list = np.arange(80,120,0.01)
    h = 0.1
    
    for i in vol_2M:
        call1 = BSM(S0, r, K_list - h, 6 / 12, i)
        call2 = BSM(S0, r, K_list, 6 / 12, i)
        call3 = BSM(S0, r, K_list + h, 6 / 12, i)

    den_2M = (call1 - 2 * call2 + call3) / h ** 2
    
    S_list = np.linspace(90,130,1000)
    call2M = call(den_2M, S_list, 100, 'call', h)
    print('2M EU call price is: ', call2M)