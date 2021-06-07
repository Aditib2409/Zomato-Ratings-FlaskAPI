import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
import os 

"""import matplotlib.ticker as mtick
plt.style.use('fivethirtyeight')"""

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

""" laoding the data """

data = pd.read_csv('E:/Python Projects/Zomato ratings using Flask/zomato.csv')

# print(data.head(5))
# print(data.shape)
# print(data.dtypes)

# missing values 
# print(data.isna().sum())

# deleting unnecessary attributes - phone
df = data.drop(['phone'], axis = 1)

# print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

# print(df.duplicated().sum())

df.dropna(how='any', inplace=True)
# print(df.isnull().sum())

# print(df.shape)

df = df.rename(columns={'approx_cost(for two people)': 'cost', 'listed_in(type)':'type', 'listed_in(city)': 'city'})
# print(df.columns)

""" cleaning the dataset """

# the cost attribute is of dtype = object. This has to be changed to int. We use the Lambda function

df['cost'] = df['cost'].apply(lambda a: a.replace(',',''))
df['cost'] = df['cost'].astype(int)

# print(df['cost'].unique())

df = df.loc[df.rate != 'NEW']
df['rate'] = df['rate'].apply(lambda a: a.replace('/5',''))

#print(df['rate'].unique())

""" VISUALIZATIONS """
X = df['book_table'].value_counts()
colors = ['#800080', '#0000A0']

pie = go.Pie(labels=X.index, values=X, textinfo="value")
fig1 = go.Figure(data = [pie])
py.iplot(fig1, filename='pie_chart_subplots')

if not os.path.exists("images"):
    os.mkdir("images")

# Restaurants that deliver online
sns.countplot(df['online_order'])
fig2 = plt.gcf()
plt.title('Whether Restaurants deliver online or Not')
plt.show()
plt.savefig('images/online_rest.png')

# Rating dustribution
sns.distplot(df['rate'])
fig3 = plt.gcf()
plt.title('Rating Distribution')
plt.show()


df['rate'] = df['rate'].astype(float)
# number of ratings between 1 and 2
a  = ((df['rate']>=1) & (df['rate']<2)).sum()
print(a)
# number of ratings between 2 and 3
b  = ((df['rate']>=2) & (df['rate']<3)).sum()
print(b)
# number of ratings between 3 and 4
c  = ((df['rate']>=3) & (df['rate']<4)).sum()
print(c)
# number of ratings between 4 and 5
d  = ((df['rate']>=4) & (df['rate']<=5)).sum()
print(d)

slice = [a,b,c,d]
colors = ['#ff3333', '#c2c2d6', '#6699ff']
labels = ['1<rate<2', '2<rate<3', '3<rate<4', '>4']
plt.pie(slice, colors=colors, labels=labels, autopct = '%1.0f%%')
fig4 = plt.gcf()
plt.title('Percentage of Restuarants according to their ratings')
plt.show()

# service types
box = sns.countplot(df['type'])
box.set_xticklabels(sns.countplot(df['type']).get_xticklabels(), rotation=30, ha="right")
fig5 = plt.gcf()
fig5.set_size_inches(4,6)
plt.title('Type of Service')
plt.show()

# distribution of cost of food for two people

data = [go.Box(y = df['cost'], name="cost distribution")]
fig = go.Figure(data=data) # define layout to adjust the size of the fig and add title
py.iplot(fig)

# MOST LIKED DISHES

import re
likes = []

for i in range(df.shape[0]):
    split_array = re.split(",",df['dish_liked'][i])
    for j in split_array:
        likes.append(j)

fav_food = pd.Series(likes).value_counts()
print(fav_food.head(10))


# Restaurant types and their counts

rest_counts = df['rest_type'].value_counts()[:15]
sns.barplot(rest_counts, rest_counts.index)
plt.title('Restaurant types')
plt.xlabel('count')
plt.show()

# Most famous restaurants

famous_rest = df['name'].value_counts()[:15]
sns.barplot(famous_rest, famous_rest.index)
plt.title('Restaurant Popularity')
plt.xlabel('count')
plt.show()

"""BUILDING THE MODEL"""

# Convert categorical data to numerical data

df.online_order[df.online_order == 'Yes'] = 1
df.online_order[df.online_order == 'No'] = 0

df.book_table[df.book_table == 'Yes'] = 1
df.book_table[df.book_table == 'No'] = 0

# label encoding of Catergorical data

from sklearn.preprocessing import LabelEncoder

df.location = LabelEncoder().fit_transform(df.location)
df.rest_type = LabelEncoder().fit_transform(df.rest_type)
df.cuisines = LabelEncoder().fit_transform(df.cuisines)
df.menu_item = LabelEncoder().fit_transform(df.menu_item)

my_data = df.iloc[:,[3,4,5,6,7,8,10,11,12]]
if not os.path.exists("E:/Python Projects/Zomato ratings using Flask/Zomato_df.csv"): 
    my_data.to_csv('E:/Python Projects/Zomato ratings using Flask/Zomato_df.csv')

x = df.iloc[:,[3,4,6,7,8,10,11,13]]
x.to_csv('E:/Python Projects/Zomato ratings using Flask/x_df.csv')
y = df['rate']


# split the dataset into train, test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 10)

R2score = []

# Linear Regression Model
model1 = LinearRegression()
model1.fit(x_train, y_train)
from sklearn.metrics import r2_score
y_pred = model1.predict(x_test)
R2score.append(r2_score(y_test, y_pred))

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor()
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
R2score.append(r2_score(y_test,y_pred))

# Random Forest
from sklearn.ensemble import RandomForestRegressor
model3 = RandomForestRegressor(n_estimators=650,random_state=245,min_samples_leaf=0.0001)
model3.fit(x_train, y_train)
y_pred = model3.predict(x_test)
R2score.append(r2_score(y_test,y_pred))

# ExtraTrees Regressor
model4 = ExtraTreesRegressor(n_estimators=119)
model4.fit(x_train, y_train)
y_pred = model4.predict(x_test)
R2score.append(r2_score(y_test,y_pred))

#print(R2score)

import pickle

pickle.dump(model4, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
