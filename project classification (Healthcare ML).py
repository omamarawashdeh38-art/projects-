#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[4]:


df = pd.read_csv(r'C:\Users\user\Downloads\healthcare_dataset.csv\healthcare_dataset.csv')


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.duplicated().sum()


# In[11]:


df.drop_duplicates(inplace=True)


# In[12]:


df.drop(['Name'],axis=1 , inplace=True)


# In[13]:


df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])


# In[14]:


df['Date of Admission'].dt.day


# In[16]:


df['Discharge Date']=pd.to_datetime(df['Discharge Date'])


# In[17]:


(df['Discharge Date'] - df['Date of Admission'])


# In[19]:


df['Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df['Stay']


# In[20]:


df.drop(['Discharge Date','Date of Admission'],axis =1 ,inplace =True)


# In[21]:


df.head()


# In[22]:


df.drop(['Room Number'] ,axis=1,inplace=True)


# In[23]:


plt.figure(figsize=(20,10))
plt.bar(df.Doctor.value_counts().sort_values(ascending=False).head(10).index,
        df.Doctor.value_counts().sort_values(ascending=False).head(10))
plt.title("Doctors")
plt.xlabel("Doctor Name")
plt.ylabel('Count')
plt.show()


# In[24]:


plt.figure(figsize=(20,10))
plt.bar(df.Hospital.value_counts().sort_values(ascending=False).head(10).index,
        df.Hospital.value_counts().sort_values(ascending=False).head(10))
plt.title("Hospital")
plt.xlabel("Hospital Name")
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# In[25]:


df.columns


# In[26]:


plt.figure(figsize=(20,10))
plt.bar(df['Test Results'].value_counts().sort_values(ascending=False).index,
        df['Test Results'].value_counts().sort_values(ascending=False))
plt.title("Results")
plt.xlabel("Results")
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# In[27]:


df.info()


# In[28]:


df.drop(['Doctor','Hospital'],axis=1,inplace=True)


# In[29]:


def split_neg_post(st):
    return st[-1]


# In[30]:


df['Blood Type_neg']= df['Blood Type'].apply(split_neg_post) 


# In[31]:


df['Blood Type_neg'].head()


# In[32]:


def split_neg_post_A(st):
    return st[:-1]


# In[33]:


df['Blood Type_P']= df['Blood Type'].apply(split_neg_post_A) 


# In[34]:


df['Blood Type_P'].head()


# In[35]:


df.drop(['Blood Type'],axis = 1 ,inplace=True)


# In[36]:


df.nunique()


# In[37]:


cols=df.select_dtypes('object')


# In[38]:


cols


# In[39]:


from sklearn.preprocessing import LabelEncoder #convert categorical data to numerical values 
for c in cols:
    lb=LabelEncoder()
    df[c]=lb.fit_transform(df[[c]])


# In[40]:


df.head()


# In[41]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[42]:


df['Billing Amount']=scaler.fit_transform(df[['Billing Amount']])


# In[43]:


df.corr()


# In[44]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)


# In[45]:


X=df.drop('Test Results',axis = 1)
y=df['Test Results']


# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


X_train , X_test ,y_train , y_test = train_test_split(X,y,test_size=0.2,shuffle=True,stratify=y)


# In[48]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Dictionary of models
models = {
    
    'LogisticRegression': LogisticRegression(max_iter = 2000),

    'RandomForestClassifier': RandomForestClassifier(),
    
    'KNeighborsClassifier' : KNeighborsClassifier(),
    
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    
    'GaussianNB'            : GaussianNB(),
    
    'Support Vector Machine' : SVC()
}


# In[49]:


from tqdm import tqdm
# Fit models, predict and calculate accuracy and F1 score
results = []
models_name = []
for name, model in tqdm(models.items()):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    models_name.append(name)
    results.append([accuracy,precision,recall,f1])


# In[50]:


Model_accuracy = pd.DataFrame(results,index=models_name,columns = ['Accuracy','Precision','Recall','F1 Score'])


# In[51]:


Model_accuracy


# In[52]:


# Plotting
Model_accuracy.plot(kind='bar', figsize=(10, 6))

# Customizing the plot

plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Model Accuracy Scores')
plt.xticks(rotation=45)  # Rotate model names for better readability
plt.legend(loc='upper right')
plt.tight_layout()  # Adjust layout to fit labels

# Display the plot
plt.show()


# In[ ]:




