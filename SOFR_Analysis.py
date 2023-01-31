# ALMASTONE CASE STUDY PART II
# SOFR TERM CONTRACT TECH EXPLORATION & ANALYSIS


# [ #1 ] - IMPORT & DOWNLOAD HISTORICAL SOFR TERM DATA VIA CME DATAMINE

import pandas as pd

# Create dataframe from csv file with SOFR Term data
df_sofr = pd.read_csv("/Users/danielpoole/ALMASTONE/case_study/csv/SOFRT_data.csv")

# Pre process SOFR term data for analysis
df_sofr = df_sofr.filter(items=['businessDate', 'rate', 'productCode'])
df_sofr = df_sofr.rename(columns={'productCode': ''})
df_sofr = df_sofr.reset_index(drop=True)
df_sofr = df_sofr.replace("T1Y", "TY1")
df_sofr['businessDate'] = pd.to_datetime(df_sofr['businessDate'])
df_sofr = df_sofr.pivot_table(index='businessDate', columns='', values='rate')

# Add column data to dataframe for spreads between different SOFR term contracts
df_sofr = df_sofr.assign(TY1_TR3=df_sofr['TY1'] - df_sofr['TR3'])
df_sofr = df_sofr.assign(TY1_TR6=df_sofr['TY1'] - df_sofr['TR6'])
df_sofr = df_sofr.assign(TR6_TR3=df_sofr['TR6'] - df_sofr['TR3'])
df_sofr = df_sofr.rename(columns={'TR1':'SOFR1M','TR3':'SOFR3M','TR6':'SOFR6M','TY1':'SOFR1Y','TY1_TR3': '1Y-3M SOFR', 'TY1_TR6': '1Y-6M SOFR', 'TR6_TR3': '6M-3M SOFR'})
df_sofr = df_sofr.drop(["SOFR1M"], axis=1)
print(df_sofr)
print("\n")


# [ #2 ] - USE MACRO ECONOMIC FACTORS & CREATE AN ML MODEL THAT FORECASTS TERM SOFR SPREADS 


# ( 2.1 ) IMPORT HISTORIAL (5Y) MACRO ECONOMIC DATA VIA FRED API

from fredapi import Fred
import datetime

# Set the FRED API key
fred = Fred(api_key='9639aebd65c8ea38eb7f8ac3e9315071')


# (2.1.1) IMPORT 10Y-2Y TREASURY YIELD SPREAD DATA, CREATE DATAFRAME & PRE PROCESS DATA FOR ANALYSIS

df_10y2y = pd.DataFrame(fred.get_series('T10Y2Y',observation_start='2020-10-01', observation_end='2023-01-05'))
df_10y2y = df_10y2y.reset_index()
df_10y2y = df_10y2y.rename(columns={'index': 'Date 10Y2Y', 0: '10Y2Y'})
df_10y2y['Date 10Y2Y'] = pd.to_datetime(df_10y2y['Date 10Y2Y'])
df_10y2y = df_10y2y.set_index('Date 10Y2Y')
df_10y2y = df_10y2y.dropna(axis= 0, how='any')
print(df_10y2y)
print("\n")



# (2.1.2) IMPORT EFFECTIVE FEDERAL FUND RATE DATA, CREATE DATAFRAME & PRE PROCESS DATA FOR ANALYSIS

df_effr = pd.DataFrame(fred.get_series('EFFR',observation_start='2020-10-01', observation_end='2023-01-05'))
df_effr = df_effr.reset_index()
df_effr = df_effr.rename(columns={'index': 'Date EFFR', 0: 'EFFR'})
df_effr['Date EFFR'] = pd.to_datetime(df_effr['Date EFFR'])
df_effr = df_effr.set_index('Date EFFR')
df_effr = df_effr.dropna(axis= 0, how='any')
print(df_effr)
print("\n")



# (2.1.3) JOIN IMPORTED DATA INTO 1 MAIN DATAFRAME FOR ANALYSIS

df = df_sofr.join(df_10y2y,how='left')
df = df.join(df_effr,how='left')
df = df.dropna(axis= 0, how='any')

# Create a new dataframe with every day within the range & interpolate for missing values
index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
df_main = pd.DataFrame(index=index, columns=df.columns)
df_main = df.reindex(index)
df_main.interpolate(method='time', axis=0, inplace=True)
df_main = df_main.dropna(axis= 0, how='any')
print(df_main)
print("\n")



# (2.1.4) DETERMINE OPTIMAL PARAMETERS p,d,q FOR ARIMA MODEL

import pmdarima as sm
from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA

# Split dataframe into predictors and targets
predictors = ['EFFR','10Y2Y']
targets = ['1Y-3M SOFR', '1Y-6M SOFR', '6M-3M SOFR']

# Fit the model
for target in targets:
    stepwise_fit = auto_arima(df_main[target], df_main[predictors], start_p=0, start_q=0,
                            max_p=3, max_q=3, m=12,
                            start_P=0, seasonal=True,
                            d=None, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
    
    p = stepwise_fit.order[0] 
    d = stepwise_fit.order[1]
    q = stepwise_fit.order[2]
    
    print(f'{target} Optimal parameter values: p={p}, d={d}, q={q}')



# (2.1.5) CREATE ARIMA MODEL AND FIT TO DATA

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split

# Split dataframe into predictors and targets
X = df_main[['EFFR','10Y2Y']]
Y = df_main[['1Y-3M SOFR', '1Y-6M SOFR', '6M-3M SOFR']]

# Split the data into a train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Create ARIMA models from training set
model_1Y3M = ARIMA(Y_train['1Y-3M SOFR'],order=(1,0,0), exog=X_train).fit()
model_1Y6M = ARIMA(Y_train['1Y-6M SOFR'],order=(1,0,0), exog=X_train).fit()
model_6M3M = ARIMA(Y_train['6M-3M SOFR'],order=(1,0,0), exog=X_train).fit()

# Make predictions on test dataset
pred_1Y3M = model_1Y3M.predict(start=X_train.shape[0],end=X.shape[0]-1, exog=X_test)
pred_1Y6M = model_1Y6M.predict(start=X_train.shape[0],end=X.shape[0]-1, exog=X_test)
pred_6M3M = model_6M3M.predict(start=X_train.shape[0],end=X.shape[0]-1, exog=X_test)

# Evaluate ARIMA Model & create a table of actual vs predicted values
comparison = pd.DataFrame({'Actual SOFR1Y3M': Y_test['1Y-3M SOFR'], 'Predicted SOFR1Y3M': pred_1Y3M,
                           'Actual SOFR1Y6M': Y_test['1Y-6M SOFR'], 'Predicted SOFR1Y6M': pred_1Y6M,
                           'Actual SOFR6M3M': Y_test['6M-3M SOFR'], 'Predicted SOFR6M3M': pred_6M3M
                           })  
print(comparison)



# (2.1.6) EVALUATE MODEL PERFORMANCE ON TEST DATASET

from sklearn import metrics
from sklearn.metrics import mean_squared_error

# Calculate the mean of each test data array
print('MEAN1Y3M:', Y_test['1Y-3M SOFR'].mean())
print('MEAN1Y6M:', Y_test['1Y-6M SOFR'].mean())
print('MEAN6M3M:', Y_test['6M-3M SOFR'].mean())

# Calculate & print error metrics MAE & MSE
print('MAE1Y3M:', metrics.mean_absolute_error(Y_test['1Y-3M SOFR'], pred_1Y3M))
print('MAE1Y6M:', metrics.mean_absolute_error(Y_test['1Y-6M SOFR'], pred_1Y6M))
print('MAE6M3M:', metrics.mean_absolute_error(Y_test['6M-3M SOFR'], pred_6M3M))
print('MSE1Y3M:', metrics.mean_squared_error(Y_test['1Y-3M SOFR'], pred_1Y3M))
print('MSE1Y6M:', metrics.mean_squared_error(Y_test['1Y-6M SOFR'], pred_1Y6M))
print('MSE6M3M:', metrics.mean_squared_error(Y_test['6M-3M SOFR'], pred_6M3M))



# (2.1.7) USE ARIMA MODELS TO PREDICT SOFR SPREADS FOR THE NEXT YEAR

# Define the number of periods to forecast
forecast_periods = 365

# Make predictions using historical data
pred_1Y3M = model_1Y3M.predict(start=df_main.shape[0],end=df_main.shape[0]+forecast_periods-1,exog=X.iloc[-531:])
pred_1Y6M = model_1Y6M.predict(start=df_main.shape[0],end=df_main.shape[0]+forecast_periods-1,exog=X.iloc[-531:])
pred_6M3M = model_6M3M.predict(start=df_main.shape[0],end=df_main.shape[0]+forecast_periods-1,exog=X.iloc[-531:])

# Create dataframe for 1Y predicted values @ daily intervals
df_forecast = pd.DataFrame({'1Y-3M SOFR':pred_1Y3M.values,'1Y-6M SOFR':pred_1Y6M.values,'6M-3M SOFR':pred_6M3M.values})
index = pd.date_range(start='2023-01-06', end='2024-01-05', freq='D')
df_forecast = df_forecast.set_index(index)
print(df_forecast)

# Create dataframe for 1Y predicted values @ monthly intervals
df_monthly = pd.DataFrame(index=pd.date_range(start='2023-01-06', end='2024-01-01', freq='M'))
df_monthly_forecast = df_monthly.join(df_forecast,how='left')
print(df_monthly_forecast)



# (2.1.8) PLOT HISTORICAL & PREDICTED VALUES VS TIME

import matplotlib.pyplot as plt
import numpy as np

# Join historical data and predicted data into one dataframe for plots
df2 = df_main[['1Y-3M SOFR', '1Y-6M SOFR', '6M-3M SOFR']]
df2 = df2.append(df_forecast)
print(df2)

# Plot 1Y-3M SOFR Term Spreads vs Time
fig, ax = plt.subplots()
index = pd.date_range(start='2020-10-01', end='2024-01-05', freq='D')
ax.set_ylim(ymin=min(df2['1Y-3M SOFR']),ymax=max(df2['1Y-3M SOFR']))
ax.plot(index,df2['1Y-3M SOFR'])
ax.set(xlabel='Time', ylabel='SOFR 1Y-3M (%)',
       title='1Y-3M SOFR Spread vs Time')
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=7)
plt.savefig('figs/1Y_3M_SOFR.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot 1Y-6M SOFR Term Spreads vs Time
fig, ax = plt.subplots()
index = pd.date_range(start='2020-10-01', end='2024-01-05', freq='D')
ax.set_ylim(ymin=min(df2['1Y-6M SOFR']),ymax=max(df2['1Y-6M SOFR']))
ax.plot(index,df2['1Y-6M SOFR'])
ax.set(xlabel='Time', ylabel='SOFR 1Y-6M (%)',
       title='1Y-6M SOFR Spread vs Time')
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=7)
plt.savefig('figs/1Y_6M_SOFR.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot 6M-1M SOFR Term Spreads vs Time
fig, ax = plt.subplots()
index = pd.date_range(start='2020-10-01', end='2024-01-05', freq='D')
ax.set_ylim(ymin=min(df2['6M-3M SOFR']),ymax=max(df2['6M-3M SOFR']))
ax.plot(index,df2['6M-3M SOFR'])
ax.set(xlabel='Time', ylabel='SOFR 6M-3M (%)',
       title='6M-3M SOFR Spread vs Time')
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=7)
plt.savefig('figs/6M_3M_SOFR.jpg', bbox_inches='tight', dpi=150)
plt.show()



# [ #3 ] - TRADING STRATEGY FOR SWAPS ACCOUNTING FOR RISK VS RETURN

import empyrical as pf
from empyrical import sharpe_ratio

#Returns of the strategy
example_returns = pd.Series([0.02, 0.03, -0.01, 0.05, 0.02])

#Risk-free rate
rf = 0.01

# Calculate Sharpe ratio
sharpe = pf.sharpe_ratio(example_returns, risk_free = rf)
print('Sharpe Ratio:',sharpe)


