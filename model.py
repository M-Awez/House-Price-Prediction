import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

class MLR:
    def __init__(self):
        df=pd.read_csv('data.csv')
        df['date']=pd.to_datetime(df['date'],dayfirst=False,errors='coerce')
        df['day']=df['date'].dt.day
        df['month']=df['date'].dt.month
        df['year']=df['date'].dt.year
        df=df.drop(['date'],axis=1)
        df['country'] = df['country'].map({'USA': 1})
        l=df['city'].unique()
        d={}
        for i in range(len(l)):
            d[l[i]] = i
        df['city'] = df['city'].map(d)
        self.X=df.iloc[:,1:]
        self.y=df.iloc[:,0:1]

    def split(self):
        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,train_size=0.8,random_state=42)
        return X_train,X_test,y_train,y_test

    def train(self,a,b):
        self.reg=LinearRegression()
        self.reg.fit(a,b)
        print("Model Trained")

    def predictions(self,a):
        return self.reg.predict(a)

    def accuracy_loss(self, y_train, prds):
        n=0
        d=0
        y_train=y_train.to_numpy().flatten()
        prds=prds.flatten()
        mean=y_train.mean()
        for i in range(len(y_train)):
            n+=(y_train[i]-prds[i]) ** 2
            d+=(y_train[i]-mean) ** 2
        acc=1-(n / d)
        print(f"Accuracy: {acc*100}")
        loss=(n/len(y_train))**(1/2)
        print(f"loss: {loss}")


obj=MLR()
X_train,X_test,y_train,y_test=obj.split()
obj.train(X_train,y_train)
train_prds=obj.predictions(X_train)
print("TRAINING DATA")
obj.accuracy_loss(y_train,train_prds)
print("TEST DATA")
test_prds=obj.predictions(X_test)
obj.accuracy_loss(y_test,test_prds)
with open("model.pkl",'wb') as f:
    pickle.dump(obj.reg,f)
print("Pickle File Created")
