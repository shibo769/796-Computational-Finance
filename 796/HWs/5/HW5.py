import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import bs4 as bs
from pandas_datareader import data as pdr
import yfinance as yf
import statsmodels.api as sm

def getdata(ticker,startdate,enddate):
    
    yf.pdr_override()
    result = pdr.get_data_yahoo(ticker,start = startdate, end = enddate)


    resultseries = pd.DataFrame(data = result['Adj Close'],index = result.index)
    #resultseries[ticker] = result['Adj Close']
    resultseries.columns = [ticker]
    return resultseries

def spy_components():
    req = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(req.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    components = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker.replace(".", "-")
        components.append(ticker.replace("\n", ""))

    return components

def clean_data(df):
    data = df
    data.ffill(axis=0, inplace=True)
    num = sum(data.isna().sum())
    print("File NaN number: %d" % num)
    return data

def data_return(df):
    adj_close_shift = df.shift(1)
    adj_ret = df / adj_close_shift - 1
    adj_ret.dropna(axis=0, how = 'all',inplace=True)
    return adj_ret

def plot(data,title,xlabel,ylabel):
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

###########################Problem1(1)###########################
spydata = getdata("SPY", '2016-02-22', '2021-02-22')
snp = spy_components()
Data = snp[0:101]
data = yf.download(Data, '2016-02-22', '2021-02-22')['Adj Close']
data = clean_data(data)
print('The number of NaN after data cleaning: \n', data.isna().sum())
###########################Problem1(2)###########################
return_data = data_return(data)
log_return = np.log(1 + return_data)
log_return = log_return.drop(log_return.tail(1).index)
del log_return['CARR']
print('A sequence of daily log returns for each asset for each date: \n'\
      ,log_return)
###########################Problem1(3)###########################
cov_return = log_return.cov()
print("Covariance matrix of daily returns: \n",cov_return)
eng_w, eng_v = np.linalg.eig(cov_return)
plot(eng_w,'Eigenvalue','','')
###########################Problem1(4)###########################
sorted_eng_w = sorted(eng_w, reverse = True)
sum50 = [sum(sorted_eng_w[:i]) >= 0.5 * sum(sorted_eng_w) for i in range(1,1 + len(sorted_eng_w))]
num50 = len(sum50) - sum(sum50)
print('The number of eigenvalues account for 50 percent of vaiance is: %d' % num50)
sum90 = [sum(sorted_eng_w[:i]) >= 0.9 * sum(sorted_eng_w) for i in range(1,1 + len(sorted_eng_w))]
num90 = len(sum90) - sum(sum90)
print('The number of eigenvalues account for 90 percent of vaiance is: %d' % num90)
###########################Problem1(5)###########################
H = np.mat(log_return);U = eng_v[:, np.argsort(eng_w)[num90:100][::-1]];HU = H * U
Y = np.mat(np.sum(log_return.values, axis=1)).T
beta = np.linalg.inv(HU.T * HU) * HU.T * Y;res = np.array(Y - HU * beta).flatten()
plot(res,"Return Stream of Residuals",'','')
# Normality of Residuals
plt.figure()
sm.qqplot(res, fit=True, line='45')
plt.show()
###########################Problem2(1)###########################(???)
R = np.mat(np.mean(log_return)).T; a = 1; C = np.mat(cov_return); c = np.matrix([[1],[0.1]])
G = np.mat([list(np.ones(100)), list(np.ones(17)) + list(np.zeros(100 - 17))])
GCG = np.linalg.inv(G * np.linalg.inv(C) * G.T)
print("Inverse of GC(-1)GT is:\n", GCG)
###########################Problem2(2)###########################(???)
Lambda = GCG * (G * np.linalg.inv(C) * R - 2 * a * c)
w = np.array(1 / (2 * a) * np.linalg.inv(C) * (R - G.T * Lambda))
plt.figure()
plt.title("Portfolio of 100 stocks")
plt.ylabel("Weight")
plt.bar([x for x in range(1, 101)], list(w.flatten()))
plt.show()