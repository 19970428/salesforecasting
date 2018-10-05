from flask import Flask, jsonify, request
import xlrd
import requests
import json
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
#import xlrd
app = Flask(__name__)
dataset = pd.read_excel("Sample.xls")
df = pd.DataFrame(dataset)
furniture = df.loc[df['Category'] == 'Furniture']
furniture.drop(df.columns.difference(['Order Date','Sales']), 1, inplace=True)
# dt = furniture['Order Date'].values.astype('int')
# sales = furniture['Sales'].values@app.route("/rf")
app = Flask(__name__)
@app.route('/rf1')
def rf1():    
    dt = furniture['Order Date'].values.astype('int')
    sales = furniture['Sales'].values       
    data = {'SalesPredictions' : list(sales),'date' : list(dt)}
    return jsonify(data)
@app.route('/abcd',methods = ['GET'])
def abcd():    
    url = "http://127.0.0.1:5000/rf1"
    try:
        uResponse = requests.get(url)
    except requests.ConnectionError:
        return "Connection Error"
    Jresponse = uResponse.text
    data = json.loads(Jresponse)
    list1 = [k for k in data['date']]
    list2 = [k for k in data['SalesPredictions']]
    
    array1 = np.asarray(list1).reshape(-1,1)
    array2 = np.asarray(list2)
    
    
    X_train,X_test,y_train,y_test = train_test_split(array1,array2,test_size=0.2)
    regressor = RandomForestRegressor()
    regressor = regressor.fit(X_test,y_test)
    sales_pred = regressor.predict(X_test)
    sales_pred = list(sales_pred)
    return jsonify(sales_pred)    
if __name__ == '__main__':
   app.run()
   
   
   
   
   
   
   
   
   
   
   
   
   