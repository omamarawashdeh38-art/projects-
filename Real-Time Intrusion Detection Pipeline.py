#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import logging
import time
import requests
import pandas as pd


# In[2]:


unseen_df = pd.read_csv(r"C:\Users\user\Documents\New folder\intrusion_unseen.csv")
unseen_df


# In[3]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

unseen_df['Timestamp'] = pd.to_datetime(unseen_df['Timestamp'], format='%d/%m/%Y %H:%M:%S')

unseen_df['Year'] = unseen_df['Timestamp'].dt.year
unseen_df['Month'] = unseen_df['Timestamp'].dt.month
unseen_df['Day'] = unseen_df['Timestamp'].dt.day
unseen_df['Hour'] = unseen_df['Timestamp'].dt.hour
unseen_df['Minute'] = unseen_df['Timestamp'].dt.minute
unseen_df['Second'] = unseen_df['Timestamp'].dt.second

unseen_df = unseen_df.drop(columns=['Timestamp'], axis=1)

exclude_labels = ['Label']
cols_to_standardize = unseen_df.columns.difference(exclude_labels)

standard_scaler = StandardScaler()
unseen_df[cols_to_standardize] = standard_scaler.fit_transform(unseen_df[cols_to_standardize])


# In[4]:


#Extract data
def extract_data(number_of_records):
    logging.info("Extracting data...")
    time.sleep(2)  
    url = 'http://87.236.232.200:5000/data'

    try:
        response = requests.get(url, params={'records': number_of_records})
        if response.status_code == 200:
            data = response.json()
            df=pd.DataFrame(data)
            df= df.reindex(sorted(df.columns), axis=1)
        else:
            logging.error(f"Failed to fetch data. Status code: {response.status_code}")

    except requests.RequestException as e:
        logging.error(f"Request failed: {e}")
    return df


# In[5]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import zscore

def transform_data(input_df):
    logging.info("Transforming data...")
    time.sleep(3)

    transformed_df = input_df.copy()

    transformed_df['Timestamp'] = pd.to_datetime(transformed_df['Timestamp'], format='%d/%m/%Y %H:%M:%S')

    transformed_df['Year'] = transformed_df['Timestamp'].dt.year
    transformed_df['Month'] = transformed_df['Timestamp'].dt.month
    transformed_df['Day'] = transformed_df['Timestamp'].dt.day
    transformed_df['Hour'] = transformed_df['Timestamp'].dt.hour
    transformed_df['Minute'] = transformed_df['Timestamp'].dt.minute
    transformed_df['Second'] = transformed_df['Timestamp'].dt.second

    transformed_df = transformed_df.drop(columns=['Timestamp'], axis=1)
    transformed_df = transformed_df.drop(['timestamp', 'source_ip'], axis=1)

    # Encoding categorical columns
    exclude_labels = ['Label']
    categorical_cols = transformed_df.select_dtypes(include=['object']).columns
    selected_cols = [col for col in categorical_cols if col not in exclude_labels]
    data_subset = transformed_df[selected_cols]

    label_encoder = LabelEncoder()
    data_subset_encoded = data_subset.apply(lambda x: label_encoder.fit_transform(x.astype(str)) if x.name in selected_cols else x)

    transformed_df[selected_cols] = data_subset_encoded

    # Z-score test for outliers
    zscore_df = transformed_df.drop(['Label'], axis=1).apply(zscore)
    zscore_outliers = transformed_df[(zscore_df > 3).any(axis=1)]

    transformed_df = transformed_df[~transformed_df.index.isin(zscore_outliers.index)]

    # Standardization using StandardScaler
    exclude_labels = ['Label']
    cols_to_standardize = transformed_df.columns.difference(exclude_labels)

    standard_scaler = StandardScaler()
    transformed_df[cols_to_standardize] = standard_scaler.fit_transform(transformed_df[cols_to_standardize])

    return transformed_df


# In[6]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Train and evaluate model function
def train_and_evaluate_model(input_data):
    logging.info("Training model...")
    time.sleep(2)
    
    # Use a subset of your transformed data for testing
    test_data = input_data.sample(frac=0.2)  # Assuming 20% of your data for testing
    X_test = test_data.drop('Label', axis=1)
    y_test = test_data['Label']

    X_train = input_data.drop('Label', axis=1)
    y_train = input_data['Label']

    param_grid = {
       'C': [0.1, 1, 10],
       'kernel': ['linear', 'rbf'],
       'gamma': ['scale', 'auto']
    }

    svc_classifier = SVC(random_state=42)
    grid_search = GridSearchCV(svc_classifier, param_grid, cv=10)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    optimized_svc_classifier = SVC(**grid_search.best_params_, random_state=42)
    optimized_svc_classifier.fit(X_train, y_train)
    
    y_pred_cv = cross_val_predict(optimized_svc_classifier, X_train, y_train, cv=10)
    classification_report_result = classification_report(y_train, y_pred_cv)
    print('Classification Report:\n', classification_report_result)

    f1_scores = f1_score(y_train, y_pred_cv, average=None)
    print('F1 Scores for Each Class:', f1_scores)
    print('Mean F1 Score:', f1_scores.mean())

    y_pred_test = optimized_svc_classifier.predict(X_test)

    classification_report_result_test = classification_report(y_test, y_pred_test)
    print('Classification Report on Test Data:\n', classification_report_result_test)





# In[ ]:


import logging
import time


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Monitoring

def monitor_pipeline():
    while True:
        try:
            extracted_data = extract_data(1000)
            transformed_data = transform_data(extracted_data)
            trained_model = train_and_evaluate_model(transformed_data)
            # Log successful execution
            logging.info("Pipeline executed successfully")

        except Exception as e:
            # Log any exceptions or errors
            logging.error(f"Pipeline execution failed: {str(e)}")

        # Wait for a certain interval before executing the pipeline again
        time.sleep(15)

# Start monitoring the pipeline
monitor_pipeline()


# In[ ]:




