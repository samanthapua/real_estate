#Using XGBoost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error

df = pd.read_csv('./Melbourne_housing_FULL.csv')
df['Date']= pd.to_datetime(df['Date'], dayfirst=True,format='%d/%m/%y')
df = df.set_index('Date')

""" Visualizing data"""
# df.plot(style=".",figsize=(15,5),title='Housing Price in Melbourne', y='Price')

"""Train/Test Split"""
train =df.loc[df.index<='31/12/2016']
test = df.loc[df.index>'31/12/2016']
# fig,ax = plt.subplots(figsize=(15,5))
# train.plot(ax=ax,label='Training Set',y='Price')
# test.plot(ax=ax,label='Test Set',y='Price')

"""Feature Creation"""
def create_features(df):
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    return df

df = create_features(df)
"""Visualize Feature / Target Relationship"""
fig,ax = plt.subplots(figsize=(15,5))
sns.boxplot(data=df, x='dayofweek',y='Price')
ax.set_title('Housing Price by Quarter')


"""Create Model"""
reg = xgb.XGBRegressor(n_estimators=1000,early_stopping_rounds=50)
train_new = create_features(train)
test_new = create_features(test)
FEATURES = ['dayofweek','quarter','month','dayofyear']
TARGET = 'Price'
X_Train = train_new[FEATURES]
Y_Train = train_new[TARGET]

X_Test = test_new[FEATURES]
Y_Test = test_new[TARGET]

reg.fit(X_Train, Y_Train,eval_set=[(X_Train,Y_Train),(X_Test,Y_Test)], verbose=True)

"""Feature Importance"""
fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh',title='Feature Importance')

"""Forecast on test"""
test['prediction']= reg.predict(X_Test)
df = df.merge(test[['prediction']],how='left',left_index=True,right_index=True)
ax = df[['Price']].plot(figsize=(15,5))
df['prediction'].plot(ax=ax,style='.')
plt.legend(['True Data','Predictions'])
ax.set_title('Raw Data and Prediction')
plt.show()


"""
18Mar - Finish tutorial
19Mar - gym + poker
20Mar - coding + swimming + gym
21Mar - office + coding
22Mar - office(?)/malaysia(?)
23Mar - bball
"""

