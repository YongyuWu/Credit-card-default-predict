#!/usr/bin/env python
# coding: utf-8

# In[17]:


from flask import Flask
import joblib


# In[18]:


app = Flask(__name__)


# In[19]:


from flask import request, render_template


# In[22]:


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method=="POST":
        income = request.form.get("Income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        print(income, age, loan)
        income_norm = (float(income) - 45136.88)/14425.49
        age_norm = (float(age) - 34.80)/12.84
        loan_norm = (float(loan) - 12.84)/3174.52
        model = joblib.load("Default")
        pred = model.predict([[float(income_norm), float(age_norm),float(loan_norm)]])
        print(pred)
        s = "The predicted default score is " + str(pred[0])
        return(render_template("creditcarddetails.html", result = s))
    else:
        return(render_template("creditcarddetails.html", result="Please enter values"))


# In[23]:


if __name__=="__main__":
    app.run()


# In[ ]:





# In[ ]:




