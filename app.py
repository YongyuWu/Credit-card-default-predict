#!/usr/bin/env python
# coding: utf-8

# In[11]:


from flask import Flask
app = Flask(__name__)


# In[12]:


from flask import request, render_template
import joblib


# In[13]:


@app.route("/",methods = ["GET", "POST"])
def i():
    if request.method == "POST":
        income = request.form.get("Income")
        age = request.form.get("age")
        loan = request.form.get("loan") 
        print(income, age, loan)
        model = joblib.load("CreditCardDefault")
        income_norm = (float(income) - 45136.875975)/14425.486619
        age_norm = (float(age) - 34.795950)/12.840055
        loan_norm = (float(loan) - 12.840055)/3174.522430
        pred = model.predict([[float(income_norm), float(age_norm),float(loan_norm)]])
        print(pred)
        s = "The predicted default is : " + str(pred)
        return(render_template("creditcarddetails.html", result = s))
    else:
        return(render_template("creditcarddetails.html", result="Please enter values"))
        


# In[14]:


if __name__=="__main__":
    app.run()


# In[ ]:





# In[ ]:




