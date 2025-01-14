#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import pandas as pd

# In[ ]:


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        credit_score = request.form["credit_score"]
        country = request.form["country"]
        age = request.form["age"]
        tenure = request.form["tenure"]
        balance = request.form["balance"]    
        products_number = request.form["products_number"]
        credit_card = request.form["credit_card"]
        active_member = request.form["active_member"]
        estimated_salary = request.form["estimated_salary"]  
        X = pd.DataFrame([[int(credit_score), str(country),
                       int(age), int(tenure), float(balance), 
                       int(float(products_number)), int(float(credit_card)), int(float(active_member)),
                       float(estimated_salary)]],columns=['credit_score','country','age','tenure','balance','products_number','credit_card','active_member','estimated_salary'])
       
        X = pd.get_dummies(X,columns=['country'], prefix=['country'])
        X = pd.get_dummies(X,columns=['products_number'], prefix=['products_no'])
        countries = ['country_Germany','country_Spain','country_France']
        ct_value = 'country_'+str(country)
        makeup_c = [x for x in countries if x != ct_value]
        products_no = ['products_no_1','products_no_2','products_no_3','products_no_4']
        pd_value = 'products_no_'+str(products_number)
        makeup_pd = [x for x in products_no if x != pd_value]
        makeup = makeup_c + makeup_pd
        for i in makeup:
           X[i] = [0]*len(X)
        X = X[['credit_score','age','tenure','balance','credit_card','active_member','estimated_salary','country_France','country_Germany','country_Spain','products_no_1','products_no_2','products_no_3','products_no_4']]
        pred = model.predict_proba(X)[0,1]
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)

