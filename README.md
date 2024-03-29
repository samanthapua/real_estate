Data Analytics Project for Real Estate

1. Data Visualization for Melbourne Housing (Easy)
    - Data exploratory on tableau and streamlit
        - Country Area against Price: Find the increment in price over time over against each region, list the regions that constantly has higher prices
        - House Area against Price: Plot rooms against price, carpark against price, landsize against price. bedrooms2-rooms(partition room) against price
    - Embed tableau dashboard on github 

2. Time Series Analysis (Average)
   - To explore and analyze/intepret patterns in time-ordered data to explain trends and seasonality. 
   - Implement time series forecasting - using statistical model to predict future values of a time-series based on past results
   - Conduct ADCF Test - test to see if data is stationary
   - ARIMA model (AR- Auto Regressive,I - Integration, MA - Moving Average)
     - AR: correlation between previous time period and current time period. P = autoregressive lags
     - I: d = order of differentiation
     - MA: Q = moving
Components in TSA:
   - Trend - general direction of time series data over a long period of time i.e. Increasing (upwards), Decreasing (downwards), Horizontal (stationary)
   - Seasonality - trend that repeats w.r.t to timing, direction and magnitude
   - Cyclical Component - no set repetition over a particular period of time
   - Irregular Variation - trend and cyclical variations are removed, variations are unpredictable, erratic and may/may not be random
   - ETS Decomposition - Error, Trend, Seasonality
Performing TSA
a. To test for stationary - conduct Dickey-Fuller test
```
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(Datasets[column_to_test_stationary],autolag='AIC')
dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','# of Lags Used','# of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'$%key]=value
print(dfoutput)
```
b. To estimate trend
```
log_df = np.log(indexDataset)
# X to be determine and same as test for stationary
movingAverage = log_df.rolling(window=X).mean()
movingSTD = log_df.rolling(window=X).std()

```
3. Time Series Forecasting with XGBoost (Hard)
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
```

4. Predictive Modeling (Hard)
Use Python with scikit-learn or TensorFlow/PyTorch to build a predictive model for housing prices.
Evaluate different regression algorithms and choose the one with the best performance.


Tools utilised:
- Tableau
- Streamlit
- Anaconda
- Pycharm
- Github
