import pandas as pd
import statsmodels.api as sm
# apply SARIMAX MODEL to all data.
path_file='201709.csv'
temp_data = pd.read_csv(path_file, index_col = 'Time')
list_val = []
for i in range(temp_data.shape[1]):
    val = 0
    for j in range(temp_data.shape[0]):
        val += abs(temp_data.iloc[j, i])
    temp = [i, val, temp_data.iloc[:,i].name]
    list_val.append(temp)
list_i = sorted(list_val, key=lambda d: d[1],reverse= True)
for i in list_i:
    print(i[0])
data = False
for i in range(100):
    index = list_i[i][0]
    try:
        mod = sm.tsa.statespace.SARIMAX(temp_data.iloc[:,index], trend='n', order=(1,0,1), seasonal_order=(1,1,1,24))
        results = mod.fit()
        fit = results.fittedvalues
        print(i)
        predict =results.predict(start = len(temp_data.iloc[:,i]), end= len(temp_data.iloc[:,i])+23)
        predict.name = list_i[i][2]
        print(predict)
    except:
        mod = sm.tsa.statespace.SARIMAX(temp_data.iloc[:, i], trend='n', order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
        results = mod.fit()
        fit = results.fittedvalues
        print(i)
        predict = results.predict(start=len(temp_data.iloc[:, i]), end=len(temp_data.iloc[:, i]) + 23)
        predict.name = list_i[i][2]
        print(predict)
    if data is False:
        data = predict
    else:
        data = pd.concat([data, predict], join='outer', axis=1)
print(data)
data.index.name = 'Time'
data.to_csv('201709_top100.csv',index = True)
print("end")
