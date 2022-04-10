#!/usr/bin/env python
# coding: utf-8

# # Cleaning and preparing the data for model traning

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing the dataset
df_train = pd.read_csv('BlackFriday_train.csv')
df_train.head()


# In[3]:


df_test = pd.read_csv('BlackFriday_test.csv')
df_test.head()


# In[4]:


##Merge both train and test data
df = df_train.append(df_test)
df.head()


# In[5]:


#Basic
df.info()


# In[6]:


df.describe()


# In[7]:


df.drop(['User_ID'],axis=1,inplace=True)


# In[8]:


df.head()


# In[9]:


pd.get_dummies(df['Gender'])


# In[10]:


#Handling categorical feature Gender
df['Gender'] = df['Gender'].map({'F' : 0, 'M':1})
df.head()


# In[11]:


#Handle feature Age
df['Age'].unique()


# In[12]:


df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})


# In[13]:


from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
 
# Encode labels in column 'species'.
df['Age']= label_encoder.fit_transform(df['Age'])
 
df['Age'].unique()


# In[14]:


df.head()


# In[15]:


df_city = pd.get_dummies(df['City_Category'],drop_first = True)


# In[16]:


df_city.head()


# In[17]:


pd.concat([df,df_city],axis = 1)
df.head()


# In[18]:


## drop city Category Feature
df.drop('City_Category',axis = 1,inplace = True)


# In[19]:


df.head()


# In[20]:


## MISSINg Values
df.isnull().sum()


# In[21]:


## focus on replacing missing values
df['Product_Category_2'].unique()


# In[22]:


df['Product_Category_2'].value_counts()


# In[23]:


## Replace the missing value with mode
df['Product_Category_2'] = df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[24]:


df['Product_Category_2'].isnull().sum()


# In[25]:


## Product_category 3 replace missing values
df['Product_Category_3'].unique()


# In[26]:


df['Product_Category_3'].value_counts()


# In[27]:


df['Product_Category_3'] = df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[28]:


df.head()


# In[29]:


df['Stay_In_Current_City_Years'].unique()


# In[30]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+',' ')


# In[31]:


df.head()


# In[32]:


df.info()


# In[33]:


## convert object into integers
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(int)
df.info()


# In[34]:


##Visualisation
sns.barplot('Age','Purchase',hue='Gender',data = df)


# In[35]:


sns.barplot('Product_Category_1','Purchase',hue='Gender',data = df)


# In[36]:


sns.barplot('Product_Category_2','Purchase',hue='Gender',data = df)


# In[37]:


sns.barplot('Product_Category_3','Purchase',hue='Gender',data = df)


# In[38]:


df.head()


# # Feature Scaling

# In[39]:


df[df['Purchase'].isnull()]


# In[ ]:




