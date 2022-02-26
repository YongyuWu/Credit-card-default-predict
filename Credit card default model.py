#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[6]:


df = pd.read_csv('Credit Card Default II (balance).csv')


# In[7]:


df


# In[8]:


df.describe()
#all x variables are continuous variables 


# In[9]:


df.isnull().sum()
# no null values 


# In[10]:


df.isna().sum()
# no NA values


# In[11]:


df.duplicated().sum()
#no duplicates


# In[12]:


print(df[df.default==0].shape[0])
#no umbalanced data observed


# In[13]:


df.boxplot()
#no outliers observed 


# In[14]:


print(sum(df['income']<0))
print(sum(df['age']<0))
print(sum(df['loan']<0))
print(sum(df['default']<0))


# In[15]:


df = df.drop(df[df.age < 0].index)
#remove age less than 0 values as we assume them to be illogical data


# In[16]:


df


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


import seaborn as sns
sns.heatmap(df.corr())


# # train test split

# In[19]:


import random
random.seed(100)


# In[20]:


y = df.loc[:, ["default"]]


# In[21]:


x = df.iloc[:, 0:3]


# In[22]:


#Normalisation
from scipy import stats
x["income"]=stats.zscore(x["income"].astype(np.float))
x["age"]=stats.zscore(x["age"].astype(np.float))
x["loan"]=stats.zscore(x["loan"].astype(np.float))
print(x)


# In[23]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)
print(X_train, X_test, Y_train, Y_test)


# # linear regression

# In[31]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# In[104]:


model = linear_model.LogisticRegression()


# In[105]:


model.fit(X_train,Y_train)


# In[106]:


pred = model.predict(X_test)


# In[32]:


from sklearn.metrics import confusion_matrix


# In[108]:


cm = confusion_matrix(Y_test, pred)


# In[109]:


print((cm[0,0] + cm[1,1])/sum(sum(cm)))
#0.9513618677042801


# In[110]:


## QQplot
import statsmodels.api as sm
from matplotlib import pyplot as plt
data = sm.datasets.longley.load()
exog = sm.add_constant(data.exog)
mod_fit = sm.OLS(data.endog, exog).fit()
res = mod_fit.resid # residuals
fig = sm.qqplot(res)
plt.show()


# In[111]:


import statsmodels.api as sm
model=sm.Logit(y,x)
result=model.fit()
print(result.summary2())


# # Decision Tree

# In[30]:


from sklearn import tree


# In[113]:


modeltree = tree.DecisionTreeClassifier()


# In[114]:


modeltree.fit(X_train,Y_train)


# In[115]:


predtree = modeltree.predict(X_test)


# In[116]:


cm_tree = confusion_matrix(Y_test,predtree)
print(cm_tree)


# In[117]:


print((cm_tree[0,0] + cm_tree[1,1])/sum(sum(cm_tree)))


# In[119]:


#finding best params using grid
import math
from sklearn.model_selection import GridSearchCV

model = tree.DecisionTreeClassifier()
grid_maxdepth = GridSearchCV(estimator = model, param_grid = dict(max_depth = [i for i in range(1, 20)]))
grid_results = grid_maxdepth.fit(x, y)
print(grid_results.best_params_)

grid_minsamplesplit = GridSearchCV(estimator = model, param_grid = dict( min_samples_split = [i for i in range(3, 20)]))
grid_results = grid_minsamplesplit.fit(x, y)
print(grid_results.best_params_)


# In[145]:


#using max_depth = 10 and min_samples_split = 4
modeltree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 4)
modeltree.fit(X_train,Y_train)
predtree = modeltree.predict(X_test)
cm_tree = confusion_matrix(Y_test,predtree)
print(cm_tree)
print((cm_tree[0,0] + cm_tree[1,1])/sum(sum(cm_tree)))

#0.9970817120622568


# In[146]:


import matplotlib.pyplot as plt

plt.subplots(figsize=(20, 10))
tree.plot_tree(modeltree, fontsize=10)


# # Random forest

# In[29]:


from sklearn.ensemble import RandomForestClassifier


# In[161]:


modelforest = RandomForestClassifier(max_depth=10, min_samples_split = 4)


# In[162]:


modelforest.fit(X_train, Y_train)
pred_rf = modelforest.predict(X_train)


# In[163]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred_rf)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

#accuracy = 0.999165623696287


# # XG boost

# In[164]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(max_depth=10, min_samples_split = 4)
model.fit(X_train, Y_train)
pred = model.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

#accuracy = 0.995136186770428


# #### The highest accuracy is obtained from the random forest classifier model

# # MLP

# In[33]:


from sklearn.neural_network import MLPClassifier


# In[34]:


model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(6,2))


# In[35]:


model.fit(X_train, Y_train)


# In[36]:


pred = model.predict(X_test)


# In[37]:


cm = confusion_matrix(Y_test,pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

#accuracy = 0.867704280155642


# In[38]:


import joblib
joblib.dump(model, "CreditCardDefault")


# In[ ]:




