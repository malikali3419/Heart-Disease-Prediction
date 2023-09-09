#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import missingno as msn
import matplotlib.pyplot as plt
import seaborn as snb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
import pickle


# In[2]:


df=pd.read_csv('heart2.csv')


# In[3]:


df.head(20).style.background_gradient("Reds")


# In[4]:


mean_for_RestingBP = df.RestingBP.mean()


# In[5]:


df.RestingBP.fillna(mean_for_RestingBP,inplace=True)


# In[6]:


df.isnull().sum()


# In[7]:


ms = msn.matrix(df)
plt.show()


# In[8]:


# Now we check that how much people is affected with heart diseases and how many are not


# In[9]:


snb.countplot(x=df.HeartDisease)


# In[10]:


df[df['MaxHR']==0]


# In[11]:


df[df['RestingBP']==0]


# In[12]:


df=df.drop(df[df['RestingBP']==0].index)


# In[13]:


# so we droped the row which has 0 values in RestingBP


# In[14]:


df.dtypes


# In[15]:


df.Age.hist()
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("Histogram of Cholestrol and Ages people")
plt.show()


# In[16]:


df.Age.plot(kind="hist")


# In[17]:


df.groupby('Sex').Sex.count().plot(kind='pie')


# In[18]:


df.drop_duplicates()


# In[19]:


label_encoder_x=LabelEncoder()


# In[20]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1:]


# In[21]:


x.ChestPainType=label_encoder_x.fit_transform(x.ChestPainType)
x.RestingECG=label_encoder_x.fit_transform(x.RestingECG)
x.ExerciseAngina=label_encoder_x.fit_transform(x.ExerciseAngina)
x.ST_Slope=label_encoder_x.fit_transform(x.ST_Slope)
x.Sex = label_encoder_x.fit_transform(x.Sex)


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x.values,y.values, test_size=0.2)


# In[23]:


LogRegression = LogisticRegression()
LogRegression.fit(x_train,y_train)
LogPred=LogRegression.predict(x_test)


# In[24]:


print(accuracy_score(y_test,LogPred))
print(precision_score(y_test,LogPred))
print(recall_score(y_test,LogPred))
print(f1_score(y_test,LogPred))


# In[25]:


LogRegression = LogisticRegression()
LogRegression.fit(x.values,y.values)


# In[26]:


model_file= "heart_model.sav"
pickle.dump(LogRegression,open(model_file,"wb"))


# In[27]:


loadmodel = pickle.load(open(model_file,'rb'))


# In[28]:


df.head()


# In[29]:


sex = input("Enter the Sex of User : ")
age = input("Enter the Age of User : ")
chestpain = input("Enter the Chest Pain Type : ")
restingBp = input("Enter the Resting Blood Pressure : ")
cholestrol = input("Enter the Cholestro : ")
fastingBs = input("Enter the Fasting Bs : ")
restingEcg = input("Enter the Resting Ecg : ")
maxHr = input("Enter the Maximum Heart Rate : ")
exerciseAngina = input("Enter the Exercise Angina : ")
oldpeak = input("Enter the Old Peak : ")
st_slope = input("Enter the St slope : ")


# In[30]:


x


# In[31]:


print(x.Sex.unique())
print(x.ChestPainType.unique())
print(x.RestingECG.unique())
print(x.ExerciseAngina.unique())
print(x.ST_Slope.unique())


# 

# In[32]:


dict_for_decode={
    'F':0,
    'M':1,
    'NAP':2,
    'ATA':1,
    'ASY':0,
    "TA":3,
    "Normal":1,
    "ST":2,
    "LVH":0,
    "Y":1,
    "N":0,
    "Flat":1,
    "Up":2,
    "Down":0
}


# In[33]:


print(dict_for_decode)


# In[34]:


input_defs = pd.DataFrame({
    "Age":age,
    "Sex":dict_for_decode.get(sex),
    "ChestPainType":dict_for_decode.get(chestpain),
    "RestingBP":restingBp,
    "Cholesterol":cholestrol,
    "FastingBS":fastingBs,
    "RestingECG":dict_for_decode.get(restingEcg),
    "MaxHR":maxHr,
    "ExerciseAngina":dict_for_decode.get(exerciseAngina),
    "Oldpeak":oldpeak,
    "ST_Slope":dict_for_decode.get(st_slope)
},index=[0])


# In[35]:


input_defs


# In[36]:


print(LogRegression.predict(input_defs))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




