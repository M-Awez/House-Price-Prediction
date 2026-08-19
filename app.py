import pandas as pd
from flask import Flask
import pickle
from flask import render_template,request
with open('model.pkl','rb') as t:
    m=pickle.load(t)
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=["POST","GET"])
def predict():
    l=list(request.form.values())
    a=[float(i) for i in l[1:]]
    date=pd.to_datetime(l[0],dayfirst=False,errors='coerce')
    a.append(date.day)
    a.append(date.month)
    a.append(date.year)
    s=m.predict([a])
    return render_template('index.html',result=float(s[0]))
if __name__=='__main__':
    app.run(debug=True)


