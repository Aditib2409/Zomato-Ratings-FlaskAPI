
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('E:/Python projects/Zomato ratings using Flask/zomato web app framework/Zomato_df.csv')
x = pd.read_csv('E:/Python projects/Zomato ratings using Flask/zomato web app framework/x_df.csv')

y = df['rate']

x.drop('Unnamed: 0', axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.30, random_state = 10)

model = ExtraTreesRegressor(n_estimators = 119)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score

import pickle
pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(y_pred)


