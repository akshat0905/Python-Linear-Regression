# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:17:12 2018

@author: Admin
"""

import pandas as pd
import numpy as np
import math
import pylab
import matplotlib.pyplot as plt

import seaborn as sns 
%matplotlib inline
import warnings # Ignores any warning
warnings.filterwarnings("ignore")

# uses Ordinary Least Squares (OLS) method
# -------------------------------------------
import statsmodels.api as sm

from sklearn.cross_validation import train_test_split
import scipy.stats as stats

import seaborn as sns




sales.shape
]
# VIF
# ---
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Feature selection
# -----------------
from sklearn.feature_selection import f_regression as fs

# </import libraries>


# read the input file
# --------------------
path="C:\\Users\\Admin\\Documents\\Python\\project\\salesdata.csv"
sales = pd.read_csv(path)
sales.head()

train=salesiloc[]


# summarize the dataset
# clearer view. removed the 1st row as it contains same info (total records)
# ------------------------------------------------------------
desc = sales.describe()
desc = desc.drop(desc.index[0]) # dropping the record count
desc

# check for NULLS, blanks and zeroes
# -------------------------------
cols = list(sales.columns)
type(cols)
print(cols)

for c in sales:
    if (len(sales[c][sales[c].isnull()])) > 0:
        print("WARNING: Column '{}' has NULL values".format(c))

    if (len(sales[c][sales[c] == 0])) > 0:
        print("WARNING: Column '{}' has value = 0".format(c))
        


_=plt.hist(sales['Item_Weight'])
plt.show()

sales.boxplot(column='Item_Weight')
sales.boxplot(column='slag')


sales.dtypes
sales.info()



#EDA
idsUnique = len(set(sales.Item_Identifier))
idsTotal = sales.shape[0]
idsDupli = idsTotal - idsUnique

print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")


#Finding the values of Item_Outlet_Sales
n=sales.count()[11]


#Dividing the given valus into train and test

train=sales.iloc[0:(n),:]

train.tail()

test=sales.iloc[(n):sales.count()[0],0:11]

test.shape
test.head()
# Univariate Analysis

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(train.Item_Outlet_Sales, bins = 25)
plt.ticklabel_format(style='plain', axis='x', scilimits=(0,1))
plt.xlabel("Item_Outlet_Sales")
plt.ylabel("Number of Sales")
plt.title("Item_Outlet_Sales Distribution")

#checking for skew

print ("Skew is:", train.Item_Outlet_Sales.skew())
print("Kurtosis: %f" % train.Item_Outlet_Sales.kurt())


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes

corr =numeric_features.corr()
corr

print(corr['Item_Outlet_Sales'].sort_values(ascending=False))


#correation matrix

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True);

sns.countplot(train.Item_Fat_Content)

sns.countplot(train.Item_Type)
plt.xticks(rotation=90)

sns.countplot(train.Outlet_Size)

sns.countplot(train.Outlet_Location_Type)

sns.countplot(train.Outlet_Type)
plt.xticks(rotation=90)

#Bivariate anaalysis

plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Weight, train["Item_Outlet_Sales"],'.', alpha = 0.3)



plt.figure(figsize=(12,7))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Visibility and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Visibility, train["Item_Outlet_Sales"],'.', alpha = 0.3)



Outlet_Establishment_Year_pivot = \
train.pivot_table(index='Outlet_Establishment_Year', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Establishment_Year")
plt.ylabel("Sqrt Item_Outlet_Sales")
plt.title("Impact of Outlet_Establishment_Year on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


Item_Fat_Content_pivot = \
train.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.median)
Item_Fat_Content_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Item_Fat_Content")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


Outlet_Identifier_pivot = \
train.pivot_table(index='Outlet_Identifier', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Identifier_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Identifier")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Identifier on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


train.pivot_table(values='Outlet_Type', columns='Outlet_Identifier',aggfunc=lambda x:x.mode())

train.pivot_table(values='Outlet_Type', columns='Outlet_Size',aggfunc=lambda x:x.mode())

Outlet_Size_pivot = \
train.pivot_table(index='Outlet_Size', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Size_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Size")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Size on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


Outlet_Type_pivot = \
train.pivot_table(index='Outlet_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Type_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Type ")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


Outlet_Location_Type_pivot = \
train.pivot_table(index='Outlet_Location_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Location_Type_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Location_Type ")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Location_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


train.pivot_table(values='Outlet_Location_Type', columns='Outlet_Type',aggfunc=lambda x:x.mode())


# Data Pre-Processing

sales.isnull().sum()/sales.shape[0]*100 #show values in percentage

#item weight

item_avg_weight = sales.pivot_table(values='Item_Weight', index='Item_Identifier')
print(item_avg_weight)


#Note: The previous ideia becomes more clear if you run this.
def impute_weight(cols):
    Weight = cols[0]
    Identifier = cols[1]
    
    if pd.isnull(Weight):
        return item_avg_weight['Item_Weight'][item_avg_weight.index == Identifier]
    else:
        return Weight
print ('Orignal #missing: %d'%sum(sales['Item_Weight'].isnull()))
sales['Item_Weight'] = sales[['Item_Weight','Item_Identifier']].apply(impute_weight,axis=1).astype(float)
print ('Final #missing: %d'%sum(sales['Item_Weight'].isnull()))
       
#Outlet size
#Import mode function:
from scipy.stats import mode
#Determing the mode for each
outlet_size_mode = sales.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())
outlet_size_mode

def impute_size_mode(cols):
    Size = cols[0]
    Type = cols[1]
    if pd.isnull(Size):
        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]
    else:
        return Size
print ('Orignal #missing: %d'%sum(sales['Outlet_Size'].isnull()))
sales['Outlet_Size'] = sales[['Outlet_Size','Outlet_Type']].apply(impute_size_mode,axis=1)
print ('Final #missing: %d'%sum(sales['Outlet_Size'].isnull()))
       


#Item Visbility
visibility_item_avg = sales.pivot_table(values='Item_Visibility', index='Item_Identifier')
print(item_avg_weight)

       
def impute_visibility_mean(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility == 0:
        return visibility_item_avg['Item_Visibility'][visibility_item_avg.index == item]
    else:
        return visibility
print ('Original #zeros: %d'%sum(sales['Item_Visibility'] == 0))
sales['Item_Visibility'] = sales[['Item_Visibility','Item_Identifier']].apply(impute_visibility_mean,axis=1).astype(float)
print ('Final #zeros: %d'%sum(sales['Item_Visibility'] == 0))


#Handling dates
       
#Remember the data is from 2013
sales['Outlet_Years'] = 2013 - sales['Outlet_Establishment_Year']
sales['Outlet_Years'].describe()       

#Get the first two characters of ID:
sales['Item_Type_Combined'] = sales['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
sales['Item_Type_Combined'] = sales['Item_Type_Combined'].map({'FD':'Food',                                                      'NC':'Non-Consumable','DR':'Drinks'})
sales['Item_Type_Combined'].value_counts()




#Handling "Item_Fat_Content"
print('Original Categories:')
print(sales['Item_Fat_Content'].value_counts())


sales['Item_Fat_Content'] = sales['Item_Fat_Content'].replace({'LF':'Low Fat',                                                      'reg':'Regular','low fat':'Low Fat'})
print('\nModified Categories:')
print(sales['Item_Fat_Content'].value_counts())

#Mark non-consumables as separate category in low_fat:
sales.loc[sales['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
sales['Item_Fat_Content'].value_counts()






#Creating dummies

#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
sales['Outlet'] = le.fit_transform(sales['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    sales[i] = le.fit_transform(sales[i])

sales = pd.get_dummies(sales, columns =['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])
sales.dtypes



sales.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:


train=sales.iloc[0:(n),:]

train.shape
train.tail()

test=sales.iloc[(n):sales.count()[0],:]

test.drop(['Item_Outlet_Sales'],axis=1,inplace=True)
test.shape
test.head()


#Export files as modified versions:
train.to_csv("C:\\Users\\Admin\\Documents\\Python\\project\\train_modified.csv",index=False)
test.to_csv("C:\\Users\\Admin\\Documents\\Python\\project\\test_modified.csv",index=False)



target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']


xx=train['Item_Outlet_Sales']

train.drop(['Item_Outlet_Sales'],axis=1,inplace=True)

train['Item_Outlet_Sales']=xx

train1, test1 = train_test_split(train, test_size = 0.3)

train1.shape
test1.shape
train_x.shape

train_x = train1.iloc[:,0:32] 
train_y = train1.iloc[:,32]
test_x  = test1.iloc[:,0:32]
test_y = test1.iloc[:,32]

train_x.drop(['Item_Identifier'],axis=1,inplace=True)
train_x.drop(['Outlet_Identifier'],axis=1,inplace=True)
test_x.drop(['Item_Identifier'],axis=1,inplace=True)
test_x.drop(['Outlet_Identifier'],axis=1,inplace=True)

train_y
train_x = sm.add_constant(train_x)
test_x = sm.add_constant(test_x)
train_y.dtypes

train_x.dtypes


lm1 = sm.OLS(train_y, train_x).fit()

lm1.summary()

pdct1 = lm1.predict(test_x)
print(pdct1)


mse = np.mean((pdct1 - test_y)**2)
print("MSE = {0}, RMSE = {1}".format(mse,math.sqrt(mse)))



#----------------------------------------------------
train_x.drop(['Item_Weight'],axis=1,inplace=True)
train_x.drop(['Item_Visibility'],axis=1,inplace=True)
test_x.drop(['Item_Weight'],axis=1,inplace=True)
test_x.drop(['Item_Visibility'],axis=1,inplace=True)



lm2 = sm.OLS(train_y, train_x).fit()

lm2.summary()

pdct2 = lm2.predict(test_x)
print(pdct2)


mse = np.mean((pdct2 - test_y)**2)
print("MSE = {0}, RMSE = {1}".format(mse,math.sqrt(mse)))



#Model number 3

train_x.drop(['Item_Fat_Content_2'],axis=1,inplace=True)
test_x.drop(['Item_Fat_Content_2'],axis=1,inplace=True)


train_x.drop(['Item_Type_Combined_0'],axis=1,inplace=True)
test_x.drop(['Item_Type_Combined_0'],axis=1,inplace=True)

train_x.drop(['Item_Type_Combined_1'],axis=1,inplace=True)
test_x.drop(['Item_Type_Combined_1'],axis=1,inplace=True)


train_x.drop(['Item_Type_Combined_1'],axis=1,inplace=True)
test_x.drop(['Item_Type_Combined_1'],axis=1,inplace=True)



lm3 = sm.OLS(train_y, train_x).fit()

lm3.summary()

pdct3 = lm3.predict(test_x)
print(pdct3)


mse = np.mean((pdct3 - test_y)**2)
print("MSE = {0}, RMSE = {1}".format(mse,math.sqrt(mse)))



# VIF (Variance Inflation Factor)
# -------------------------------
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(train_x.values, i) for i in range(train_x.shape[1])]
vif["features"] = train_x.columns
print(vif)





### validating the assumptions
# ----------------------------
def getresiduals(lm,train_x,train_y):
    predicted = lm.predict(train_x)
    actual = train_y
    residual = actual-predicted
    
    return(residual)
    


train_x.shape
train_y.shape
residuals = getresiduals(lm1,train_x,train_y)
print(residuals)

# 1) Residual mean is 0
# ----------------------------
print(residuals.mean())

# 2) Residuals have constant variance
# ------------------------------------
y = lm1.predict(train_x)
sns.set(style="whitegrid")
sns.residplot(residuals,y,lowess=True,color="g")

# 3) Residuals are normally distributed
# --------------------------------------
stats.probplot(residuals,dist="norm",plot=pylab)
pylab.show()

# 4) rows > columns
# ------------------
conc.shape





















#------------------------------------------


from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(train_x,train_y) 
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)
new=clf.predict(test_x)
print(test_y,new)