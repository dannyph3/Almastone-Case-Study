# ALMASTONE CASE STUDY PART I
# SLC AGRICOLA FINANCIAL ANALYSIS



# [ #1 ] - IMPORT & DOWNLOAD FINANCIAL DATA FOR SLC AGRICOLA

import yfinance as yf
import pandas as pd

# Get data for SLC Agricola, Soybeans, Corn & Cotton Futures
symbols = ["SLCJY", "ZS=F", "ZC=F", "CT=F", "6L=F"]

for x in symbols:
    
    # Get the stock data for SLC Agricola
    ticker = yf.Ticker(x)

    # Get the historical data for the stock
    historical_data = ticker.history(period="5y", interval="1d")

    # Select the columns we want and rename them
    data = historical_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.rename(columns={'Open': 'Daily Open', 'High': 'Daily High', 'Low': 'Daily Low', 'Close': 'Daily Close', 'Volume': 'Volume'})

    # Save the data to a CSV file in directory
    data.to_csv('csv/' + x + ".csv")



# [ #2 ] - EXPLORE RELATIONSHIP BETWEEN STOCK PRICE, COMMODITY FUTURES & FX RATE


# (2.1) CREATE CENTRAL DATAFRAME & PROCESS DATA FOR ANALYSIS

# Create dataframes from csv files for SLC Stock Price + Soybean, Corn, Cotton, & Brazilian Real Futures 
df0 = pd.read_csv("/Users/danielpoole/ALMASTONE/case_study/csv/SLCJY.csv")
df1 = pd.read_csv("/Users/danielpoole/ALMASTONE/case_study/csv/ZS=F.csv")
df2 = pd.read_csv("/Users/danielpoole/ALMASTONE/case_study/csv/ZC=F.csv")
df3 = pd.read_csv("/Users/danielpoole/ALMASTONE/case_study/csv/CT=F.csv")
df4 = pd.read_csv("/Users/danielpoole/ALMASTONE/case_study/csv/6L=F.csv")

# Join above dataframes into one main dataframe for analysis
df =[]
df = df0.join(df1, how = 'left', rsuffix = ' ZS')
df = df.join(df2, how = 'left', rsuffix = ' ZC')
df = df.join(df3, how = 'left', rsuffix = ' CT')
df = df.join(df4, how = 'left', rsuffix = ' 6L')

# Filter main dataframe for Daily Close columns only
df = df.filter(items=['Date', 'Daily Close', 'Daily Close ZS', 'Daily Close ZC', 'Daily Close CT', 'Daily Close 6L'])
df = df.rename(columns={'Daily Close': 'SLC Price', 'Daily Close ZS': 'Soy Price', 'Daily Close ZC': 'Corn Price', 'Daily Close CT': 'Cotton Price', 'Daily Close 6L': 'BRL Rate'})
df = df.dropna(axis= 0, how='any')
df['Date'] = pd.to_datetime(df['Date'])

# Print central dataframe
print("\n")
print("SLC STOCK & COMMODITY DATAFRAME")
print("\n")
print(df)

# Print central dataframe key metrics
print("\n")
print("Dataframe Overview - Key Metrics:")
print("\n")
print(df.describe())
print("\n")


# (2.2) PLOT SLC STOCK PRICE VS COMMODITY FUTURES & FX RATE

import matplotlib.pyplot as plt
import numpy as np

# Plot SLC Stock Price vs Soybean Futures with a line of best fit
X = df['Soy Price']
Y = df['SLC Price']
fig, ax = plt.subplots()
ax.scatter(X,Y)
ax.set(xlabel='Soybean Future Contracts ($)', ylabel='SLC Agriculture Stock Price ($)',
       title='SLC Stock Price vs Soybean Futures')
coefficients = np.polyfit(X, Y, 1)
x_values_line = np.linspace(min(X), max(X), 100)
y_values_line = coefficients[0] * x_values_line + coefficients[1]
plt.plot(x_values_line, y_values_line, color="red", label='line of best fit')
ax.legend()
plt.savefig('figs/Stock vs Soybeans.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot SLC Stock Price vs Corn Futures with a line of best fit
X = df['Corn Price']
Y = df['SLC Price']
fig, ax = plt.subplots()
ax.scatter(X,Y)
ax.set(xlabel='Corn Future Contracts ($)', ylabel='SLC Agriculture Stock Price ($)',
       title='SLC Stock Price vs Corn Futures')
coefficients = np.polyfit(X, Y, 1)
x_values_line = np.linspace(min(X), max(X), 100)
y_values_line = coefficients[0] * x_values_line + coefficients[1]
plt.plot(x_values_line, y_values_line, color="red", label='line of best fit')
ax.legend()
plt.savefig('figs/Stock vs Corn.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot SLC Stock Price vs Cotton Futures with a line of best fit
X = df['Cotton Price']
Y = df['SLC Price']
fig, ax = plt.subplots()
ax.scatter(X,Y)
ax.set(xlabel='Cotton Future Contracts ($)', ylabel='SLC Agriculture Stock Price ($)',
       title='SLC Stock Price vs Cotton Futures')
coefficients = np.polyfit(X, Y, 1)
x_values_line = np.linspace(min(X), max(X), 100)
y_values_line = coefficients[0] * x_values_line + coefficients[1]
plt.plot(x_values_line, y_values_line, color="red", label='line of best fit')
ax.legend()
plt.savefig('figs/Stock vs Cotton.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot SLC Stock Price vs Brazilian Real Futures with a line of best fit
X = df['BRL Rate']
Y = df['SLC Price']
fig, ax = plt.subplots()
ax.scatter(X,Y)
ax.set(xlabel='Brazilian Real Future Contracts ($)', ylabel='SLC Agriculture Stock Price ($)',
       title='SLC Stock Price vs Brazilian Real Futures')
coefficients = np.polyfit(X, Y, 1)
x_values_line = np.linspace(min(X), max(X), 100)
y_values_line = coefficients[0] * x_values_line + coefficients[1]
plt.plot(x_values_line, y_values_line, color="red", label='line of best fit')
ax.legend()
plt.savefig('figs/Stock vs BRL.jpg', bbox_inches='tight', dpi=150)
plt.show()


# (2.3) PEARSONS COEFFICIENT CORRELATION ANALYSIS FOR STOCK PRICE VS COMMODITY FUTURES & FX RATE

from scipy.stats import pearsonr

# Calculate Pearson's Coefficient for SLC Stock Price vs Soybean Futures
r1, p_value = pearsonr(df['SLC Price'], df['Soy Price'])

# Calculate Pearson's Coefficient for SLC Stock Price vs Corn Futures
r2, p_value = pearsonr(df['SLC Price'], df['Corn Price'])

# Calculate Pearson's Coefficient for SLC Stock Price vs Cotton Futures
r3, p_value = pearsonr(df['SLC Price'], df['Cotton Price'])

# Calculate Pearson's Coefficient for SLC Stock Price vs Brazilian Real Futures
r4, p_value = pearsonr(df['SLC Price'], df['BRL Rate'])

# Create dataframe with Pearson Coefficients calculated above
pr = pd.DataFrame({'Future Contracts':['Soybean', 'Corn', 'Cotton', 'BRL'], 'Coefficients':[r1, r2, r3, r4]})

print(f"PEARSON'S CORRELATION ANALYSIS")
print("\n")
print(pr)
print("\n")


# (2.4) LINEAR REGRESSION ANALYSIS FOR SLC STOCK PRICE VS COMMODITY FUTURES

from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Divide dataset into features and predictions
X = df[['Soy Price', 'Corn Price', 'Cotton Price']] 
Y = df['SLC Price']

# Divide data into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)  

# Create linear regression model based on training data
reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)

# Extract coefficients the model found for each independent variable
attributes_coefficients = pd.DataFrame(reg.coef_, X.columns, columns=['Coefficient'])  
print("LINEAR REGRESSION ANALYSIS")
print("\n")
print("Linear Regression Coefficients:")
print("\n")
print(attributes_coefficients)
print("\n")

# Test performance of regression model on test set
Y_pred = reg.predict(X_test)
comparison = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
print("Linear Regression Actual vs Predicted Values:")
print("\n")
print(comparison)
print("\n")

# Evaluate performance of regression model
print("Evaluate Regression Model Performance:")
print("\n")
print('MAE:', metrics.mean_absolute_error(Y_test, Y_pred))  
print('MSE:', metrics.mean_squared_error(Y_test, Y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('R-SQUARED:', r2_score(Y_test, Y_pred))
print('Mean:',Y_test.mean())
print("\n")



# [ #3 ] - ANALYZE SLC AGRICOLA FINANCIAL PERFORMANCE IN RECENT YEARS


# (3.1) IMPORT QUARTERLY FINANCIAL DATA & PROCESS FOR ANALYSIS

# Create dataframe for SLC Quarterly Financial Data
qf = pd.read_csv("/Users/danielpoole/ALMASTONE/case_study/csv/SLCJY_quarterly_financials.csv")

# Preprocess data in dataframe for analysis
qf = qf.drop(["ttm","name"], axis=1)
qf = qf.reindex(columns=list(qf)[::-1])
qf = qf.dropna(axis= 0, how='any')
qf_transposed = qf.transpose()
qf_subset = qf_transposed.iloc[35:55]
qf_subset = qf_subset.replace(to_replace=",",value="",regex=True).apply(pd.to_numeric)
qf_subset.apply(pd.to_numeric)
qf_subset = qf_subset.rename(columns={0: 'totalRevenue', 2: 'costofRevenue', 3: 'grossProfit', 4: 'operatingExpense'})
qf_subset = qf_subset.filter(items=['totalRevenue', 'costofRevenue', 'grossProfit', 'operatingExpense'])

# Calculate & add a z-score column for grossProfit in dataframe
mean = qf_subset['grossProfit'].mean()
std = qf_subset['grossProfit'].std()
qf_subset['grossProfit_zscore'] = (qf_subset['grossProfit'] - mean) / std

# Create & set an index for the dataframe
index_ = pd.date_range('12-31-2017 00:00', periods = 20, freq ='Q')
qf_subset.index = index_
print("SLC QUARTERLY FINANCIAL DATA")
print("\n")
print(qf_subset)
print("\n")


# (3.2) CREATE SEVERAL CHARTS THAT HIGHLIGHT FINANCIAL PERFORMANCE

# Plot Total Revenue vs Time
fig, ax = plt.subplots()
ax.plot(index_,qf_subset['totalRevenue'])
ax.set(xlabel='Time', ylabel='Total Revenue ($)',
       title='SLC Total Revenue vs Time')
plt.savefig('figs/SLC Total Rev.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot Cost of Revenue vs Time
fig, ax = plt.subplots()
ax.plot(index_,qf_subset['costofRevenue'])
ax.set(xlabel='Time', ylabel='Cost of Revenue ($)',
       title='SLC Cost of Revenue vs Time')
plt.savefig('figs/SLC Cost of Rev.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot Gross Profit vs Time
fig, ax = plt.subplots()
ax.plot(index_,qf_subset['grossProfit'])
ax.set(xlabel='Time', ylabel='Gross Profit',
       title='SLC Gross Profit vs Time')
plt.savefig('figs/SLC Gross Profit.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot Operating Expenses vs Time
fig, ax = plt.subplots()
ax.plot(index_,qf_subset['operatingExpense'])
ax.set(xlabel='Time', ylabel='Operating Expenses',
       title='SLC Operating Expenses vs Time')
plt.savefig('figs/SLC Op Expenses.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot Gross Profit Z-score vs Time
fig, ax = plt.subplots()
ax.plot(index_,qf_subset['grossProfit_zscore'])
ax.axhline(y=0, color='r', linestyle='-')
ax.set(xlabel='Time', ylabel='Gross Profit Z-score',
       title='Gross Profit Z-score vs Time')
plt.savefig('figs/SLC Gross Profit Zscore.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot SLC Gross Profit & Soybean Futures vs Time
fig, ax = plt.subplots()
ax.plot(index_,qf_subset['grossProfit'], label = 'Gross Profit', color = 'black')
ax.set(xlabel='Time', ylabel='SLC Gross Profit ($)',
       title='SLC Gross Profit & Soybean Futures vs Time')
ax2 = ax.twinx()
ax2.plot(df['Date'],df['Soy Price'], label = 'Soybean Futures', color = 'red')
ax2.set(ylabel='Soybean Futures ($)')
ax.legend()
ax2.legend(loc='lower right')
plt.savefig('figs/Profit v Soybeans.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot SLC Gross Profit & Corn Futures vs Time
fig, ax = plt.subplots()
ax.plot(index_,qf_subset['grossProfit'], label = 'Gross Profit', color = 'black')
ax.set(xlabel='Time', ylabel='SLC Gross Profit ($)',
       title='SLC Gross Profit & Corn Futures vs Time')
ax2 = ax.twinx()
ax2.plot(df['Date'],df['Corn Price'], label = 'Corn Futures', color = 'red')
ax2.set(ylabel='Corn Futures ($)')
ax.legend()
ax2.legend(loc='lower right')
plt.savefig('figs/Profit v Corn.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Plot SLC Gross Profit & Cotton Futures vs Time
fig, ax = plt.subplots()
ax.plot(index_,qf_subset['grossProfit'], label = 'Gross Profit', color = 'black')
ax.set(xlabel='Time', ylabel='SLC Gross Profit ($)',
       title='SLC Gross Profit & Cotton Futures vs Time')
ax2 = ax.twinx()
ax2.plot(df['Date'],df['Cotton Price'], label = 'Cotton Futures', color = 'red')
ax2.set(ylabel='Cotton Futures ($)')
ax.legend()
ax2.legend(loc='lower right')
plt.savefig('figs/Profit v Cotton.jpg', bbox_inches='tight', dpi=150)
plt.show()