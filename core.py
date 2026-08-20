import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

class MLR:
    def __init__(self):
        try:
            df=pd.read_csv('Z:\ML Viharatech Projects\Mini Project MLR\data.csv')
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
        except Exception as e:
            er_type,er_msg,er_line=sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")
            
    def split(self):
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.8,
                                                                                    random_state=42)
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            er_type,er_msg,er_line=sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")

    def train(self):
        try:
            self.reg=LinearRegression()
            self.reg.fit(self.X_train,self.y_train)
            train_prds=self.reg.predict(self.X_train)
            return train_prds
        except Exception as e:
            er_type,er_msg,er_line=sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")

    def test(self):
        try:
            test_prds = self.reg.predict(self.X_test)
            return test_prds
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")

    def accuracy_loss(self,a,prds):
        try:
            n=0
            d=0
            a=a.to_numpy().flatten()
            prds=prds.flatten()
            mean=a.mean()
            for i in range(len(a)):
                n+=(a[i]-prds[i]) ** 2
                d+=(a[i]-mean) ** 2
            acc=1-(n / d)
            print(f"Accuracy: {acc*100}")
            loss=(n/len(a))**(1/2)
            print(f"loss: {loss}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")

    def custom_inputs(self):
        try:
            print(f"Result for custom Input:{self.reg.predict([[3,1.5,1340,7912,1.5,0,0,3,1340,0,1955,2005,22,0,5,7,2015]])[0][0]}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")

    def saving_the_file(self):
        try:
            with open ("model.pkl",'wb') as f:
                pickle.dump(self.reg,f)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")

if __name__ == "__main__":
    obj=MLR()
    X_train,X_test,y_train,y_test = obj.split()
    train_prds=obj.train()
    test_prds=obj.test()
    obj.accuracy_loss(y_train,train_prds)
    obj.accuracy_loss(y_test,test_prds)
    obj.custom_inputs()
    obj.saving_the_file()

