#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# The problem statement revolves around building a machine learning solution for credit default prediction. The goal is to predict credit default, categorized as 1 (default) or 0 (non-default), based on various client attributes.

# # Dataset Download Link

# https://online.stat.psu.edu/stat857/sites/onlinecourses.science.psu.edu.stat857/files/german_credit/index.csv

# # Importing Packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the Data

# In[2]:


data = pd.read_csv("D:\\Self Made Learning Projects\\Classification\\Default Credit\\German Credit Data.csv")


# # Understanding the Data

# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


data.info()


# In[6]:


data.head()


# In[7]:


data.describe()


# # Data Preprocessing

# In[8]:


data.nunique()


# In[9]:


data.isnull().sum()


# # Exploratory Data Analysis

# In[10]:


data.hist(figsize=(25,40))
plt.show()


# In[11]:


sns.countplot(x='Creditability', data = data)
plt.show()


# In[12]:


sns.countplot(x="Duration of Credit (month)",hue="Creditability", data=data)
plt.show()


# In[13]:


gender_based = data.groupby(["Sex & Marital Status","Creditability"])["Purpose"].value_counts()
gender_based


# In[14]:


sns.countplot(x="Sex & Marital Status",hue="Occupation", data=data)
plt.show()


# In[15]:


sns.countplot(x="Creditability",hue="Type of apartment", data=data)
plt.show()


# # Correlation

# In[16]:


corr = data.corr()
sns.heatmap(corr)
plt.show()


# In[17]:


sns.heatmap(corr[(corr>=0.5)])
plt.show()


# Credit Account and Duration of Credit are highly correlated values

# ## Dataset Preparation

# In[18]:


X = data.iloc[:,data.columns!="Creditability"]
Y = data.iloc[:,data.columns =="Creditability"]


# In[19]:


X


# In[20]:


Y


# # Spliting the dataset for Training and Testing

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,
                                                    test_size= 0.3, 
                                                    random_state=10)


# In[23]:


X_test


# In[24]:


X_train


# In[25]:


Y_test


# In[26]:


Y_train


# # Training the Model

# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


lg = LogisticRegression()


# In[29]:


lg.fit(X_train,Y_train)


# # Predictions

# In[30]:


prediction = lg.predict(X_test)
prediction


# # Model Evaluations

# In[31]:


from sklearn import metrics


# In[32]:


cnf_matrix = metrics.confusion_matrix(prediction, Y_test)
cnf_matrix


# True Positive = 178
# True Negative = 40
# False Positive = 24
# False Negative = 58 

# In[33]:


sns.heatmap(pd.DataFrame(cnf_matrix))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[34]:


Accuracy = metrics.accuracy_score(Y_test,prediction)
Accuracy


# In[35]:


Precision = metrics.precision_score(Y_test,prediction)
Precision


# In[36]:


Recall = metrics.recall_score(Y_test,prediction)
Recall


# # ROC Curve

# In[37]:


Probability  = lg.predict_proba(X_test)[::,-1]
false_pr, true_pr, thresholds = metrics.roc_curve(Y_test, Probability)
AUC = metrics.auc(false_pr, true_pr)
plt.plot(false_pr, true_pr, color='darkorange', lw=2, label=f'AUC = {AUC:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# # K-Fold Cross Validation

# In[38]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[50]:


kfold = KFold(n_splits=3, random_state=10, shuffle=True)
result = cross_val_score(lg,X_train,Y_train,cv=kfold, scoring='accuracy')


# In[53]:


result.mean()


# # Hyper Parameter Tuning 

# In[54]:


from sklearn.model_selection import GridSearchCV


# ### Defining Parameter Grid

# In[66]:


dual = [True, False]
max_iter = [100, 110, 120, 130, 140]
l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9]
param_grid = dict(dual = dual, max_iter=max_iter, l1_ratio=l1_ratio)


# ### Grid Search CV

# In[59]:


import time
lr = LogisticRegression(penalty='elasticnet', solver='saga')
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=3, n_jobs=1)
start_time = time.time()
grid_result = grid.fit(X_train, Y_train)


# In[60]:


print("Best Parameters: ", grid_result.best_params_)
print("Best Accuracy: ", grid_result.best_score_)
print("Time taken: %s seconds" % (time.time() - start_time))


# ### Randomized SearchCV

# In[64]:


from sklearn.model_selection import RandomizedSearchCV
random = RandomizedSearchCV(estimator=lr, param_distributions=param_grid, cv=3, n_jobs=1)
start_time = time.time()
random_result = random.fit(X_train, Y_train)


# In[65]:


print("Best Parameters: ", random_result.best_params_)
print("Best Accuracy: ", random_result.best_score_)
print("Time taken: %s seconds" % (time.time() - start_time))

