#!/usr/bin/env python
# coding: utf-8

#  

# # PART 1
# 

# In[143]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # setting seaborn default for plots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[144]:


file_path = r'C:\Users\shath\OneDrive\Documents\mining\Scoring-Dataset-26 ..csv'
data = pd.read_csv(file_path)


# In[145]:


gender_mapping = {'M': 0, 'F': 1, 'Non-Binary': 2}

data['Gender_Numeric'] = data['Gender'].map(gender_mapping)


# In[146]:


mar_mapping = {'M': 0, 'S': 1, 'Non-Binary': 2}
data['martial_status_numric'] = data['Marital_Status'].map(mar_mapping)

activity_mapping = {
    'Seldom': 1,
    'Frequent': 2,
    'Regular': 3
}

data['Website_Activity_Numeric'] = data['Website_Activity'].map(activity_mapping)

#print(data[['Website_Activity', 'Website_Activity_Numeric']])


# In[147]:


Browsed_Electronics_12Mo={'Yes':1,'No':0,'Non-Binary': 2}
data['Browsed_Electronics_12Mo_num']=data['Browsed_Electronics_12Mo'].map(Browsed_Electronics_12Mo)


# In[148]:


Bought_Electronics_12Mo={'Yes':1,'No':0,'Non-Binary': 2}
data['Bought_Electronics_12Mo_num']=data['Bought_Electronics_12Mo'].map(Bought_Electronics_12Mo)


# In[149]:


Bought_Digital_Media_18Mo_map={'Yes':1,'No':0,'Non-Binary': 2}
data['Bought_Digital_Media_18Mo_num']=data['Bought_Digital_Media_18Mo'].map(Bought_Digital_Media_18Mo_map)


# In[150]:


Bought_Digital_Books_map={'Yes':1,'No':0,'Non-Binary': 2}
data['Bought_Digital_Books_num']=data['Bought_Digital_Books'].map(Bought_Digital_Books_map)


# In[151]:


activity_mapping = {
    "'Bank Transfer'": 0,
    "'Credit Card'": 1,
    "'Monthly Billing'": 2,
    "'Website Account'": 3
}


data['p'] = data['Payment_Method'].map(activity_mapping)


# In[152]:


# Print the updated DataFrame
#print(data[['Payment_Method', 'p']])
# Display the result
print(data)


# # PART 2

# # B1. 
# Use the following binning techniques to smooth the values of the "Age" attribute:
#  equi-width binning (3 bins).
#  equi-depth binning (3 bins) 

# In[153]:


file_path = r'C:\Users\shath\OneDrive\Documents\mining\Scoring-Dataset-26 ..csv'
df = pd.read_csv(file_path)


# In[154]:


bins_num= 3

#Equal width binning

df['Age_equi_width'] = pd.cut(df['Age'], bins=bins_num, labels=False)
#Equal depth binning

df['Age_equi_depth'] = pd.qcut(df['Age'], q=bins_num, labels=False)

df.to_csv('B.1.csv', index=False)

df.head()


# In[155]:


df['Age_equi_width'] = pd.cut(df['Age'], bins=bins_num)

#Equal depth binning

df['Age_equi_depth'] = pd.qcut(df['Age'], q=bins_num)

print(df[['Age', 'Age_equi_width', 'Age_equi_depth']])


# In[156]:


print("Unique Age values:", df['Age'].unique())
print("Equal-width bin boundaries:", pd.cut(df['Age'], bins=bins_num).unique())
print("Equal-depth bin boundaries:", pd.qcut(df['Age'], q=bins_num).unique())


# # B.2  
# Use the following techniques to normalise the "Age" attribute:
#  min-max normalization to transform the values onto the range [0.0-1.0].
#  z-score normalization to transform the values.
# 

# In[157]:


# Min-Max Normalization
scaler_min_max = MinMaxScaler()
df['Age_min_max'] = scaler_min_max.fit_transform(df[['Age']])


# Z-Score Normalization
scaler_z_score = StandardScaler()
df['Age_z_score'] = scaler_z_score.fit_transform(df[['Age']])
#df['Age_z_score'] = df['Age_z_score'].clip(0, None)  # Set lower bound to 0


print(df[['Age', 'Age_min_max', 'Age_z_score']].head())

df.to_csv('B.2.csv', index=False)
#df
df


# # B.3
# Discretise the Age attribute into the following categories: 
#  Teenager = 1-16; 
#  Young = 17-35; 
#  Mid_Age = 36-55; 
#  Mature = 56-70; 
#  Old = 71+. 
# Provide the frequency of each category in your data set.

# In[158]:


bins = [0, 16, 35, 55, 70, float('inf')]
labels = ['Teenager', 'Young', 'Mid_Age', 'Mature', 'Old']
df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)   #The right=False parameter means that the intervals are left-closed.

# the frequency of each category
categ_count = df['Age_Category'].value_counts()

print("Frequency of Age Categories:")
print(categ_count)
df.to_csv('B.3.csv', index=False)
df


# # B.4
# Convert the "Gender" variable into binary variables [with values "0" or "1"].

# In[159]:


df = pd.get_dummies(df, columns=['Gender'], drop_first=True) #This is because if you have a dummy variable for "Male" (0 or 1), the information about being "Female" is already contained in the absence of the "Male" dummy

df.to_csv('B.4.csv', index=False)
df


# In[ ]:





# # PART 3

# In[160]:


#pip install mlxtend


# In[161]:


file_path = r"C:\Users\shath\OneDrive\Documents\mining\Community-Participation-DataSet(26).csv"
dataset = pd.read_csv(file_path)


# In[162]:


dataset.shape


# In[163]:


dataset.info()


# In[164]:


Q1 = dataset['Age'].quantile(0.25)
Q3 = dataset['Age'].quantile(0.75)
IQR = Q3 - Q1

# Determine the upper and lower bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count the number of outliers
outliers = ( dataset['Age']< lower_bound) | (dataset['Age'] > upper_bound)
num_outliers = outliers.sum()

print(f"Number of outliers: {num_outliers}")


# In[165]:


sns.boxplot(data=dataset, y='Age')
plt.ylabel('Age')
plt.title('Box Plot of Age with Outliers')
plt.show()


# In[166]:


Q1 = dataset['Elapsed_Time'].quantile(0.25)
Q3 = dataset['Elapsed_Time'].quantile(0.75)
IQR = Q3 - Q1

# Determine the upper and lower bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count the number of outliers
outliers = ( dataset['Elapsed_Time']< lower_bound) | (dataset['Elapsed_Time'] > upper_bound)
num_outliers = outliers.sum()

print(f"Number of outliers: {num_outliers}")


# In[167]:


sns.boxplot(data=dataset, y='Elapsed_Time')
plt.ylabel('Elapsed_Time')
plt.title('Box Plot of Elapsed_Time with Outliers')
plt.show()


# In[168]:


dataset['Age'].max()


# In[169]:


dataset['Age'].min()


# In[170]:


# Define the age bins and labels
bins = [17, 30, 45, 58]  # Adjust the bin edges as needed
labels = ['young', 'adult', 'older']

# Create a new column 'AgeGroup' with the discretized age information
dataset['AgeGroup'] = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)

# Display the DataFrame with the new 'AgeGroup' column


# In[171]:


df=dataset.copy()


# In[172]:


df.drop('Age',axis = 1,inplace=True)


# In[173]:


df.drop('Elapsed_Time',axis = 1,inplace=True)


# In[174]:


df.drop('Record#',axis = 1,inplace=True)


# In[175]:


df


# In[176]:


df.info()


# In[177]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Select columns for analysis
columns_for_analysis = ['Time_in_Community', 'Gender', 'Working', 'Family', 'Hobbies',
                         'Social_Club', 'Political', 'Professional', 'Religious', 'Support_Group', 'AgeGroup']

# Create a new DataFrame with selected columns
df_selected = df[columns_for_analysis]

# Convert the dataset into a list of transactions
transactions = df_selected.values.tolist()

# Convert the transactions into a one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)


# In[178]:


rules


# In[ ]:





# In[ ]:





# In[ ]:




