import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,6

data = pd.read_csv('./Melbourne_housing_FULL.csv')
data['Date'] = pd.to_datetime(data['Date'])
indexedData = data.set_index(['Date'])
print(data)
