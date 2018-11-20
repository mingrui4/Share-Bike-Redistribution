import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
# try to fit SARIMAX MODEL with a particular station
def date_trans(i):
    date = pd.datetime.strptime(i, '%m/%d/%Y %H:%M:%S')
    return date
day = 6
test = pd.read_csv('test.csv', usecols=['starttime','stoptime','start station id','end station id'])
test['null'] = 0
station = test.groupby(['start station id'])['null'].sum()
# station.name = 'time'
test['starttime'] = test['starttime'].apply(date_trans)
test['stoptime'] = test['stoptime'].apply(date_trans)
n = 16 * 24
start = datetime.datetime(year=2016, month=4,day=1,hour=0,minute=0,second=0)
end = datetime.datetime(year=2016, month=4,day=1,hour=1,minute=0,second=0)
data = False
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

for i in range(n):
# print(startsta.head(100))
    start = start + datetime.timedelta(hours = 1)
    end = end + datetime.timedelta(hours = 1)
    temp_start = test[(start<test['starttime'])&(test['starttime']<end)]
    temp_stop = test[(start<test['stoptime'])&(test['stoptime']<end)]
    start_count = station + temp_start.groupby(['start station id'])['starttime'].count()
# start_count.name = 'time'
    start_count.fillna(value=0, inplace=True)
    # print(start_count)
    stop_count = station + temp_stop.groupby(['end station id'])['stoptime'].count()
# stop_count.name = 'time'
    stop_count.fillna(value=0, inplace=True)
    temp_count = - start_count + stop_count
    temp_count.name = start
    if data is False:
        data = temp_count
    else:
        data = pd.concat([data, temp_count], join='outer', axis=1)
# print(data.T.head())
# plt.plot(data.T['72'])
# plt.show()
temp_data = data.transpose()
# test_stationarity(temp_data.iloc[:,0])
decomposition = seasonal_decompose(temp_data.iloc[:,0])
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
diff_data = temp_data.iloc[:,2] - temp_data.iloc[:,2].shift(1)
sea_data = diff_data-diff_data.shift(12)
test_stationarity(sea_data.dropna(inplace=False))
plt.subplot(411)
plt.plot(temp_data.iloc[:,0], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label="What's Left: Residuals")
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# print(temp_start.groupby(['start station id'])['starttime'].count())
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sea_data.iloc[25:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sea_data.iloc[25:], lags=40, ax=ax2)
plt.show()
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(diff_data.iloc[2:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(diff_data.iloc[2:], lags=40, ax=ax2)
plt.show()
mod = sm.tsa.statespace.SARIMAX(temp_data.iloc[:,1], trend='n', order=(1,0,1), seasonal_order=(2,1,2,24))
# mod = sm.tsa.ARMA(temp_data.iloc[:,0],order=(0,0))
results = mod.fit()
# print(results.summary())
# predict = results.predict(start = 176, end= 188, dynamic= True)
print(results.predict(start = 175, end= 187, dynamic= True))
fit = results.fittedvalues
# plt.plot(predict)
plt.plot(fit)
plt.plot(temp_data.iloc[:,1])
plt.plot(results.predict(start = len(temp_data.iloc[:,1]), end= len(temp_data.iloc[:,1])+24, dynamic= True))
plt.title('RMSE: %.4f'% np.sqrt(sum((fit-temp_data.iloc[:,1])**2)/len(temp_data.iloc[:,1])))
plt.show()