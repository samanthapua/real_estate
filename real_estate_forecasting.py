#Using XGBoost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


df = pd.read_csv('./Melbourne_housing_FULL.csv')
df = df.set_index('Date')
print(df.head())
""" Visualizing data"""
df.plot(style=".",figsize=(15,5),title='Housing Price in Melbourne', y='Price')

"""Train/Test Split"""
train =df.loc[df.index<='31/12/2016']
test = df.loc[df.index>'31/12/2016']

fig,ax = plt.subplots(figsize=(15,5))
train.plot(ax=ax,label='Training Set',y='Price')
test.plot(ax=ax,label='Test Set',y='Price')
plt.show()

"""Feature Creation"""
18Mar - Finish tutorial
19Mar - gym + poker
20Mar - coding + swimming + gym
21Mar - office + coding
22Mar - office(?)/malaysia(?)
23Mar - bball

