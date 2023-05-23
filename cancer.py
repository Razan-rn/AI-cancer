#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("cansser.csv")
df.head()


# In[2]:


df.info()


# In[3]:


df.shape


# In[4]:


df.drop(['id','Unnamed: 32'],axis=1,inplace=True)


# In[5]:


df.isnull().sum()


# In[6]:


X=df.drop(columns='diagnosis')
Y=df['diagnosis']


# In[7]:


# from sklearn.preprocessing import LabelEncoder
# lb=LabelEncoder()
# Y=lb.fit_transform(Y)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 4, shuffle = True)


# In[9]:


from sklearn.svm import SVC
def svm_model(X_train,Y_train,X_test):
    svm = SVC(kernel='linear', C=1,random_state=0)
    svm.fit(X_train,Y_train)
    Y_pred=svm.predict(X_test)
    return Y_pred


# In[10]:


Y_pred=svm_model(X_train,Y_train,X_test)
Y_pred


# In[11]:


from sklearn.metrics import accuracy_score
accuracy_sklearn = accuracy_score(Y_test, Y_pred)*100
print('Model Accuracy:',accuracy_sklearn)


# In[12]:


from sklearn.linear_model import LogisticRegression
def LR_model(X_train,Y_train,X_test):
    LR = LogisticRegression()
    LR.fit(X_train,Y_train)
    Y_pred=LR.predict(X_test)
    return Y_pred


# In[13]:


Y_pred=LR_model(X_train,Y_train,X_test)
Y_pred


# In[14]:


from sklearn.metrics import accuracy_score
accuracy_sklearn = accuracy_score(Y_test, Y_pred)*100
print('Model Accuracy:',accuracy_sklearn)


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
def knn_model(X_train,Y_train,X_test,k):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    Y_pred=knn.predict(X_test)
    return Y_pred


# In[16]:


Y_pred=knn_model(X_train,Y_train,X_test,k=4)
Y_pred


# In[17]:


from sklearn.metrics import accuracy_score
accuracy_sklearn = accuracy_score(Y_test, Y_pred)*100
print('Model Accuracy:',accuracy_sklearn)


# In[18]:


k_value=range(1,20)
accuracy=[]
for k in k_value:
    Y_predict=knn_model(X_train,Y_train,X_test,k)
    accur=accuracy_score(Y_test,Y_predict)
    accuracy.append(accur)


# In[19]:


plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(k_value,accuracy,c='g')
plt.show()

