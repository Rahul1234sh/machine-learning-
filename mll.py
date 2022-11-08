#importing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,precision_score

import warnings
warnings.filterwarnings("ignore")

#Loding dataset
Xbox_data=pd.read_csv('train.csv')

#Explonatory data analysis
Xbox_data.head()

Xbox_data.describe()

Xbox_data.info()

Xbox_data['query'].nunique()

Xbox_data['query'].nunique()

Xbox_data['category'].nunique()

values = Xbox_data['sku'].value_counts()
values.head(10)

newxbox_data=Xbox_data[(Xbox_data['sku']==9854804) | (Xbox_data['sku']== 2107458)]
newxbox_data

sns.set_style('darkgrid')
sns.heatmap(newxbox_data.isnull(),cmap='viridis')

#Datapreprocessing
#including query_length

newxbox_data['query_length']=newxbox_data['query'].apply(len)
newxbox_data['query_length'].plot(bins=50, kind='hist') 

X = newxbox_data.drop(['user', 'sku', 'category'],axis=1)
y = newxbox_data.sku
X

#converting into timestamp
from datetime import datetime

X.click_time = X[['click_time']].transform(lambda x: x + '.' + '0' * 6 if len(x) == 19  else x + (26 - len(x)) * '0')
X.query_time = X[['query_time']].transform(lambda x: x + '.' + '0' * 6 if len(x) == 19  else x + (26 - len(x)) * '0')

X = X.transform({'query': (lambda x: x.lower()), 'click_time':(lambda time: int(''.join(c for c in time if c.isdigit()))), \
                'query_time': (lambda time: int(''.join(c for c in time if c.isdigit())))})
#hash based encoding

query_type = X['query'].unique()
query_size = X['query'].nunique()
query_dict = {query_type[i]: i for i in range(query_size)}
query_dict['ncaa 2011']

X =  X.transform({'query': (lambda x: query_dict[x]), 'click_time': (lambda x: x), 'query_time' : (lambda x: x)})
X['query_length']=newxbox_data['query'].apply(len)
X

#classification algorithms
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
accuracies = dict()

rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
predictions=rfc.predict(X_test)

accuracy=accuracy_score(y_test,predictions)
print(accuracy)
accuracies['Random Forest']=accuracy*100

#Logistic Regression

mnb=LogisticRegression()
mnb.fit(X_train,y_train)

prednb=mnb.predict(X_test)
accuracy=accuracy_score(y_test,prednb)
print(accuracy)
accuracies['Logistic Regression']=accuracy*100

#LightGBM
X_train_lgb = X_train.astype('float64')
y_train_lgb = y_train.astype('float64')
X_test_lgb = X_test.astype('float64')
y_test_lgb = y_test.astype('float64')

lgb = LGBMClassifier()
lgb.fit(X_train_lgb,y_train_lgb)

Y_pred = lgb.predict(X_test_lgb)
predictions = [round(value) for value in Y_pred]

accuracy = accuracy_score(y_test_lgb,Y_pred)
accuracies['lightGBM']=accuracy*100
print(accuracy)

#KNN

knn=KNeighborsClassifier(n_neighbors=35)
knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
accuracy=accuracy_score(y_test,pred_knn)
print(accuracy)
accuracies['KNN']=accuracy*100

error_rate = []

for i in range(1,50):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
Text(0, 0.5, 'Error Rate')
print(accuracies)
accu = pd.DataFrame.from_dict(accuracies,orient='index',columns = ['Accuracy'])
accu
plt.figure(figsize=(12,10))
plt.yticks(np.arange(0,100,5))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=accu.index,y=accu['Accuracy'])

#OneHotEncoding

X_hut=newxbox_data.drop(['sku','category','user'],axis=1)
query=pd.get_dummies(X_hut['query'],drop_first=True)
from datetime import datetime

X_hut.click_time = X_hut[['click_time']].transform(lambda x: x + '.' + '0' * 6 if len(x) == 19  else x + (26 - len(x)) * '0')
X_hut.query_time = X_hut[['query_time']].transform(lambda x: x + '.' + '0' * 6 if len(x) == 19  else x + (26 - len(x)) * '0')

X_hut = X_hut.transform({'click_time':(lambda time: int(''.join(c for c in time if c.isdigit()))), \
                'query_time': (lambda time: int(''.join(c for c in time if c.isdigit())))})
X_hut =  X_hut.transform({ 'click_time': (lambda x: x), 'query_time' : (lambda x: x)})
X_hut=pd.concat([X_hut,query],axis=1)
X_hut.head(10)

X_hut.info()

X_hut
y_hut=newxbox_data['sku']

Xhut_train, Xhut_test, yhut_train, yhut_test = train_test_split(X_hut, y_hut, test_size=0.3, random_state=101)
enc_accuracies=dict()

rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(Xhut_train,yhut_train)

predictions_hut=rfc.predict(Xhut_test)

print(accuracy_score(yhut_test,predictions_hut))


enc_accuracies['Random Forest']=accuracy*100

knn=KNeighborsClassifier(n_neighbors=31)
knn.fit(Xhut_train,yhut_train)
accuracy=knn.predict(Xhut_test)
print(accuracy_score(yhut_test,accuracy))

enc_accuracies['KNN']=accuracy*100

error_rate = []

# Will take some time
for i in range(1,50):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xhut_train,yhut_train)
    pred_i = knn.predict(Xhut_test)
    error_rate.append(np.mean(pred_i != yhut_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
Text(0, 0.5, 'Error Rate')

Xhut_train_lgb = Xhut_train.astype('float64')
yhut_train_lgb = yhut_train.astype('float64')
Xhut_test_lgb = Xhut_test.astype('float64')
yhut_test_lgb = yhut_test.astype('float64')

lgb = LGBMClassifier()
lgb.fit(Xhut_train_lgb,yhut_train_lgb)

Yhut_pred = lgb.predict(Xhut_test_lgb)
predictions = [round(value) for value in Yhut_pred]

accuracy = accuracy_score(yhut_test_lgb,Yhut_pred)
accuracies['lightGBM']=accuracy*100

print(accuracy)

enc_accuracies['LightGBM']=accuracy*100

mnb=LogisticRegression()
mnb.fit(Xhut_train,yhut_train)

pred_nb=mnb.predict(Xhut_test)
accuracy=accuracy_score(yhut_test,pred_nb)
print(accuracy)
accuracies['Logistic Regression']=accuracy*100

enc_accuracies['Logistic Regression']=accuracy*100
0.5149521531100478

print(accuracies)
enc_accu = pd.DataFrame.from_dict(enc_accuracies,orient='index',columns = ['Accuracy'])
enc_accu
plt.figure(figsize=(12,10))
plt.yticks(np.arange(0,100,5))
plt.title('Accuracies after onehot encoding')
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=enc_accu.index,y=accu['Accuracy'])