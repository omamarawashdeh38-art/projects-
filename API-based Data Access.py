#!/usr/bin/env python
# coding: utf-8

# # Task 1: API-based Data Access 
# 

# ## Use API to retrieve all indicators info and save  them into a data frame. Print all indicators that contain the word ‘traffic

# In[1]:


import requests
import pandas as pd

# define the api endpoint with a filter for indicators containing 'traffic' in their name
api_endpoint = "https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,'traffic')"

# Send a get request to the api endpoint defined
response = requests.get(api_endpoint)

# we convert the response data to JSON format
data = response.json()

# in this code we get the 'value' field from the JSON data or assign an empty list if value is not present
indicators_data = data.get('value', [])

# we reate a DataFrame from the retrieved data to display result
indicators_df = pd.DataFrame(indicators_data)

# here we filter indicators containing 'traffic' in their name we should becareful it's case insensitive
traffic_indicators = indicators_df[indicators_df['IndicatorName'].str.contains('traffic', case=False)]

traffic_indicators





# ## Select health indicators of interest from the available datasets. Retrieve a minimum of two datasets focusing on specific metrics such as disease prevalence, vaccination rates, mortality rates, etc. The Global Health Observatory provides a list of indicators accessible at https://www.who.int/data/gho/data/indicators. The API supports file formats in JSON.

# In[49]:


import requests

#first we define the URLs for the two api endpoints
First_URL = "https://ghoapi.azureedge.net/api/NUTRITION_ANAEMIA_CHILDREN_NUM"
Second_URL = "https://ghoapi.azureedge.net/api/NUTRITION_ANAEMIA_CHILDREN_Prev"

# the we send a get requests to the first and second URLs
indicator_for_First = requests.get(First_URL)
indicator_for_Second = requests.get(Second_URL)

# then convert the JSON responses to python dictionaries
First_data = indicator_for_First.json()
Second_data = indicator_for_Second.json()


# ## Develop Python code to initiate API requests for the retrieval of the chosen health indicators. Utilize the requests library for effective handling of HTTP requests. Save the extracted datasets as DataFrames.
# 

# In[48]:


url_data = "https://ghoapi.azureedge.net/api/NUTRITION_ANAEMIA_CHILDREN_NUM"
r = requests.get (url_data)
json_dataa = r. json()[ 'value']
df_1 = pd.json_normalize(json_dataa)
df_1


# In[6]:


url_data = "https://ghoapi.azureedge.net/api/NUTRITION_ANAEMIA_CHILDREN_Prev"
r = requests.get (url_data)
json_dataa = r. json()[ 'value']
df_2 = pd.json_normalize(json_dataa)
df_2


# ## here we display the columns from the retrived data

# In[7]:


import requests
import pandas as pd

url_data = "https://ghoapi.azureedge.net/api/NUTRITION_ANAEMIA_CHILDREN_NUM"
r = requests.get(url_data)
json_dataa = r.json()['value']
dfc_1 = pd.json_normalize(json_dataa)

print("Column Headers:")
print(dfc_1.columns)



# In[8]:


import requests
import pandas as pd

url_data = "https://ghoapi.azureedge.net/api/NUTRITION_ANAEMIA_CHILDREN_Prev"
r = requests.get(url_data)
json_dataa = r.json()['value']
dfc_2 = pd.json_normalize(json_dataa)

print("Column Headers:")
print(dfc_2.columns)


# ## here we filter data according to needed coloumns 

# In[9]:


import requests
import pandas as pd

# Define the URL for the api endpoint
url_data = "https://ghoapi.azureedge.net/api/NUTRITION_ANAEMIA_CHILDREN_NUM"

# Send a get request to the api endpoint and retrieve JSON data
r = requests.get(url_data)
json_data = r.json()['value']

# convert JSON data to a pandas dataFrame using pd.json_normalize
df = pd.json_normalize(json_data)

print("Column Headers:")
print(df.columns)

# detect the columns needed to keep in the dataFrame
filtered_columns = ['ParentLocationCode', 'ParentLocation', 'IndicatorCode']  

filtered_df = df[filtered_columns]
filtered_df


# In[47]:


import requests
import pandas as pd

url_data = "https://ghoapi.azureedge.net/api/NUTRITION_ANAEMIA_CHILDREN_Prev"
r = requests.get(url_data)
json_dataa = r.json()['value']
df = pd.json_normalize(json_dataa)
   
print("Column Headers:")
print(df.columns)
filtered_columns = ['ParentLocationCode', 'ParentLocation', 'IndicatorCode']  
filtered_df = df[filtered_columns]

filtered_df


# ### here we filter the data as the needed attribute in th parentslocationcode column for the first data data to show the EMR data related unlike above we filtered for needed data

# In[10]:


import requests
import pandas as pd

# here we get the filtered url of parentslocationcode which is EMR for the First data
url = "https://ghoapi.azureedge.net/api/NUTRITION_ANAEMIA_CHILDREN_NUM?$filter=ParentLocationCode%20eq%20%27EMR%27"

# get data from the URL and convert the response to JSON format
url_filter = requests.get(url).json()

# here we extract the value field from the JSON response containing the filtered data
data_filtered = url_filter['value']

data_df = pd.DataFrame.from_dict(data_filtered)
data_df


# ### here we filter the data as the needed attribute in the parentslocationcode column for the second data to show the AFR data related

# In[50]:


import requests
import pandas as pd
# here we get the filtered url of parentslocationcode which is AFR for the second data
url = "https://ghoapi.azureedge.net/api/NUTRITION_ANAEMIA_CHILDREN_Prev?$filter=ParentLocationCode%20eq%20%27AFR%27"

# get data from the URL and convert the response to JSON format
url_filter = requests.get(url).json()

# here we extract the value field from the JSON response containing the filtered data
data_filtered = url_filter['value']

# Convert the filtered data to a Pandas DataFrame
data_df = pd.DataFrame.from_dict(data_filtered)
data_df


# ## Perform necessary data cleaning and transformation procedures to ensure uniformity and compatibility. This involves addressing missing values and eliminating irrelevant features. Apply a minimum of three tasks related to cleaning and transforming the data 
# 
# ### here we start to clean data 

# In[26]:


# to find the sum of null columns to be removed from dataframe 1 and see missing values that need to be removed or replaced if possible
print(df_1.shape)
df_1.isnull().sum()


# In[12]:


# to find the sum of null columns to be removed from dataframe 2 and see missing values that need to be removed or replaced if possible
print(df_2.shape)
df_2.isnull().sum()


# ## Apply a minimum of three tasks related to cleaning and transforming the data 
# 
# ### here we apply 3 cleaning methods

# In[ ]:


import pandas as pd



# lis of columns to drop have null or zero values for all
drop_list = ['Dim2Type', 'Dim2', 'Dim3Type', 'Dim3', 'DataSourceDimType', 'DataSourceDim', 'Comments']
df_cleaned = df_1.drop(drop_list, axis=1)

# this removes duplicate rows
df_cleaned = df_cleaned.drop_duplicates()

#  cleaning method drops rows with any missing values from the columns kept
df_cleaned = df_cleaned.dropna()

print("Cleaned DataFrame:")
df_cleaned.head(10)


# ## Same process for the second dataframe

# In[45]:


import pandas as pd



# lis of columns to drop have null or zero values for all
drop_list = ['Dim2Type', 'Dim2', 'Dim3Type', 'Dim3', 'DataSourceDimType', 'DataSourceDim', 'Comments']
df_cleaned_2 = df_2.drop(drop_list, axis=1)

# this removes duplicate rows
df_cleaned_2 = df_cleaned_2.drop_duplicates()

#  cleaning method drops rows with any missing values from the columns kept
df_cleaned_2 = df_cleaned_2.dropna()

print("Cleaned DataFrame:")
df_cleaned.head(10)


# ## Examine the retrieved data from various sources and identify distinct attributes that can serve as criteria for grouping the datasets. Ensure that for each dataset, there is at least one identified grouping attribute
# 
# ### here we group by parentslocation from the first data and then display the dataframe 

# In[40]:


grouped_data_df=df_1.groupby('ParentLocation')
grouped_data_df.first()


# ## Examine the retrieved data from various sources and identify distinct attributes that can serve as criteria for grouping the datasets. Ensure that for each dataset, there is at least one identified grouping attribute
# 
# ### here we group by parentslocation from the second data and then display the dataframe 

# In[28]:


grouped_data_df_2=df_2.groupby('ParentLocation')
grouped_data_df_2.first()


# ## 3 Integrate the datasets into a unified dataset, addressing any potential data quality issues that may arise during the integration process
# ### here we merge the 2 dataframes into 1 dataframe using pd.merge function to easily work with in the database

# In[29]:


merged_df = pd.merge(df_1, df_2, on=['SpatialDim', 'TimeDim', 'ParentLocation'])

merged_df.head()


# # Task 2: Loading Data and Executing Queries in MySQL Database Server
# 

# ## Write Python code to create a new database within the MySQL server using the established connection.

# In[16]:


# Import the MySQL connector library
import mysql.connector

# Database connection information
user = "root"
password = "momo_swa1607"
host = "localhost"

# establish a connection to the MySQL server using the mysql.connector.connect function then fill with my credentials
connection = mysql.connector.connect(
    user=user,
    password=password,
    host=host
)

# Create a cursor to execute SQL queries
cursor = connection.cursor()

new_database_name = "Project"

# execute an SQL query to create the new database if it doesn't already exist
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {new_database_name}")
print(f"Database '{new_database_name}' created successfully")

# close the cursor and connection when done with them
cursor.close()
connection.close()





# ## Utilize the create_engine method in Python to establish a connection to the MySQL database installed on your localhost and implement Python code to load the previously integrated dataset (from Task 1) into the newly created MySQL database. Ensure that the data is appropriately mapped to the database schema.
# 
# 

# In[30]:


from sqlalchemy import create_engine

# to create a connection to the MySQL database using SQLAlchemy's create_engine function
# and we replace 'root', 'momo_swa1607', 'localhost', and 'Project' with my credentials and database name
engine = create_engine('mysql+pymysql://root:momo_swa1607@localhost/Project')

table_name = 'p'

# here we use SQLAlchemy's to_sql method to write the dataFrame (merged_df) to the specified table in the database
# where the 'if_exists' parameter is set to 'replace', which it means if the table already exists it will be replaced with the new data given
# the 'index' parameter is set to false to avoid writing dataFrame index as a separate column in the database which can lead to errors
merged_df.to_sql(table_name, con=engine, if_exists='replace', index=False)


# ## to show that every thing gone correct and there's no problem and the table is fully created

# In[33]:


# Import necessary libraries
from sqlalchemy import create_engine, Table, MetaData
import pandas as pd

# here we replace 'root', 'momo_swa1607', 'localhost', and 'Project' with my credentials and database name
engine = create_engine('mysql+pymysql://root:momo_swa1607@localhost/Project')

table_name = 'p'

# here we created a metadata object to store information about the database schema where it's easier
metadata = MetaData()

# this command reflects here the existing databas schema into the metadata 
metadata.reflect(engine)

# access the specified table in the metadata
table = Table(table_name, metadata, autoload_with=engine)

# Construct a SELECT query for the specified table
query = table.select()

# here we used SQLAlchemy to execute the select query and fetch the results into a Pandas DataFrame
data = pd.read_sql(query, engine)

data.head(10)


# ## Craft and execute a minimum of two queries of your choice on the uploaded datasets within the MySQL database. This could include selection queries, filtering, or aggregation operations.
# 

# ### here we select the distinct values of any needed column like parentslocation here it displays all attributes it contains 

# In[34]:


import mysql.connector
from mysql.connector import Error

# Replace with the database credentials
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'momo_swa1607',
    'database': 'Project'
}

# Initialize variables
connection = None
cursor = None

try:
    connection = mysql.connector.connect(**db_config)

    if connection.is_connected():
        cursor = connection.cursor()

        #prints the Distinct Values of parentslocation for example to make sure that the merge is completed and can print values
        distinct_query = "SELECT DISTINCT ParentLocation FROM Project1;"
        cursor.execute(distinct_query)
        distinct_result = cursor.fetchall()
        print("Distinct Values Query Result:")
        print(distinct_result)
#to cheack for any errors and prints what's the error
except Error as e:
    print(f"Error: {e}")
#the following close the cursor and closes the connection
finally:
    try:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
            print("\nMySQL connection closed.")
    # the followig checks if any error occured during the connection closing 
    except Error as e:
        print(f"Error during cleanup: {e}")


# ## here we did 2 queries selection and filtering the selection we selected to print 5 results where parents location = "Eastern Mediterranean" and we used the filtering query to filter and print 5 results when SpatialDim ="AFG" and the TimeDim > 2000

# In[42]:


import mysql.connector
import pandas as pd

# Replace with the database credentials
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'momo_swa1607',
    'database': 'Project'
}

# Initialize variables
connection = None
cursor = None

try:
    # Establish a connection to the database
    connection = mysql.connector.connect(**db_config)
    # we check if connected to the database then 
    if connection.is_connected():
        cursor = connection.cursor()
        # do selection query where the selcted column is parentlocation where the location is in Eastern Mediterranean
        additional_select_query = "SELECT * FROM Project1 WHERE ParentLocation = 'Eastern Mediterranean' LIMIT 5;"
        cursor.execute(additional_select_query)
        additional_select_result = cursor.fetchall()
        print("Additional Selection Query Result (First 5):")
        for row in additional_select_result:
            print(row)
        # filter query where the filter prints the SpatialDim = AFG and the Timedim>2000 it prints only 5 to prove correct
        additional_filter_query = "SELECT * FROM Project1 WHERE SpatialDim = 'AFG' AND TimeDim > 2000 LIMIT 5;"
        cursor.execute(additional_filter_query)
        additional_filter_result = cursor.fetchall()
        print("\nAdditional Filtering Query Result (First 5):")
        for row in additional_filter_result:
            print(row)
#to find what's the error if occured 
except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    # Close the cursor and connection
    if cursor:
        cursor.close()
    if connection and connection.is_connected():
        connection.close()

