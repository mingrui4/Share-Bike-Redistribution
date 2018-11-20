import pandas as pd
import datetime
import matplotlib.pyplot as plt
# data transforming
def date_trans(i):
    date = pd.datetime.strptime(i, '%Y-%m-%d %H:%M:%S')
    return date
day = 6
path_file='201709-citibike-tripdata.csv'
data1 = pd.read_csv(path_file, usecols=['starttime', 'stoptime', 'start station id', 'end station id'])
data1['null'] = 0
station = data1.groupby(['start station id'])['null'].sum()
data1['starttime'] = data1['starttime'].apply(date_trans)
data1['stoptime'] = data1['stoptime'].apply(date_trans)
n = 16 * 24
start_day=1
start_hour=0
start_minute=0
strat_second=0
end_day=1
end_hour=1
end_minute=0
end_second=0
start = datetime.datetime(year=2017, month=9,day=start_day,hour=start_hour,minute=start_minute,second=strat_second)
end = datetime.datetime(year=2017, month=9,day=end_day,hour=end_hour,minute=end_minute,second=end_second)
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
    start = start + datetime.timedelta(hours = 1)
    end = end + datetime.timedelta(hours = 1)
    temp_start = data1[(start < data1['starttime']) & (data1['starttime'] < end)]
    temp_stop = data1[(start < data1['stoptime']) & (data1['stoptime'] < end)]
    start_count = station + temp_start.groupby(['start station id'])['starttime'].count()
    start_count.fillna(value=0, inplace=True)
    stop_count = station + temp_stop.groupby(['end station id'])['stoptime'].count()
    stop_count.fillna(value=0, inplace=True)
    temp_count = - start_count + stop_count
    temp_count.name = start
    if data is False:
        data = temp_count
    else:
        data = pd.concat([data, temp_count], join='outer', axis=1)
temp_data = data.transpose()
temp_data.index.name = 'Time'
temp_data.to_csv('201709.csv',index = True)

data2 = pd.read_csv(path_file, usecols=['start station id', 'start station latitude', 'start station longitude','start station name'])
new = data2.drop_duplicates()
new.to_csv('201709_station.csv',index = True)