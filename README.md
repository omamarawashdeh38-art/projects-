# Project 1 (Digital Image Processing Projects)

This repository contains a collection of Python scripts demonstrating fundamental **Digital Image Processing (DIP)** techniques using the Pillow library.

##  Key Features
*   **Image Composition:** Techniques for merging, cropping, and rearranging image segments from different sources (e.g., Lena and Peppers images).
*   **Spatial Transformations:** Manipulating image quadrants and halves to create custom layouts.
*   **Image Negative:** Implementing pixel-wise transformations to invert colors/grayscale values ($255 - \text{pixel value}$).
*   **Noise Reduction:** Utilizing **Median Filtering** ($3 \times 3$ kernel) to effectively remove "Salt and Pepper" noise while preserving edges.

##  Tech Stack
*   **Language:** Python 3
*   **Core Library:** [Pillow (PIL)](https://python-pillow.org)

##  Featured Operations
1.  **Blending:** Creating a composite image from two distinct grayscale images.
2.  **Geometric Rearrangement:** Systematic swapping of image regions.
3.  **Restoration:** Converting noisy inputs and applying spatial filters to improve visual quality.
---------------------------------------------------------------------------------------------------------------------------
# project 2 (API-based Data Access)
### WHO Global Health Data Pipeline using REST APIs and MySQL
## Overview
This project implements an end-to-end **data engineering pipeline** that retrieves global health indicators from the **WHO Global Health Observatory (GHO) API**, cleans and integrates multiple datasets, and loads them into a **MySQL database** for structured querying and analysis.

The project focuses on **child anaemia indicators**, including absolute counts and prevalence rates.

## Key Features
- REST API data retrieval using Python  
- JSON normalization into structured Pandas DataFrames  
- Data cleaning (null handling, deduplication, feature reduction)  
- Dataset integration across time, region, and location dimensions  
- Automated loading into MySQL using SQLAlchemy  
- SQL-based data querying and filtering  

## Tech Stack
- **Python** (Pandas, Requests)
- **SQLAlchemy**
- **MySQL**
- **Jupyter Notebook**
- **WHO GHO API**

## Data Sources
- WHO Global Health Observatory API  
  - `NUTRITION_ANAEMIA_CHILDREN_NUM`  
  - `NUTRITION_ANAEMIA_CHILDREN_Prev`  

## Data Processing
- Removed irrelevant and fully-null attributes  
- Eliminated duplicate records  
- Dropped rows with missing values  
- Filtered data by WHO regions (e.g., EMR, AFR)  
- Merged datasets using spatial and temporal keys
- 
## Database & Queries
- Programmatic MySQL database creation  
- Integrated dataset loaded as a relational table  
- Example queries:
  - Regional data selection  
  - Country-level filtering by year  
  - DISTINCT region extraction  
----------------------------------------------------------------------------------------------------------------------

# Project 3 (Information Retrieval (IR) – Text Preprocessing & Feature Extraction)

## Overview
This project implements a complete **Information Retrieval (IR) pipeline** for textual data preprocessing and feature extraction.  
The system prepares raw text data for retrieval and machine learning tasks by applying cleaning, normalization, and multiple vectorization techniques.

## Key Features
- Loading ARFF text datasets into Pandas DataFrames
- Text cleaning (punctuation and digit removal)
- Stopword removal using NLTK
- Text normalization using stemming
- Feature extraction using:
  - Bag of Words (BoW)
  - TF-IDF
  - Word2Vec embeddings
- Exporting processed features for further analysis

## IR Techniques Implemented
### Text Preprocessing
- Tokenization
- Stopword removal
- Stemming (Porter Stemmer)
- Noise removal

### Feature Representation
- **Bag of Words (CountVectorizer)**
- **TF-IDF Vectorization**
- **Word2Vec (Average Word Embeddings)**
- 
## Tech Stack
- Python
- Pandas
- NLTK
- Scikit-learn
- Gensim
- NumPy

## Output
- Clean and normalized textual corpus
- TF-IDF feature matrix (`tfidf.csv`)
- Numerical vector representations using Word2Vec

## Use Case
- Information Retrieval systems
- Text similarity and search engines
- Text classification and NLP-based analysis
-----------------------------------------------------------------------------------------------------------------------------
# Project 4 (NLP Project – Extractive Text Summarization)

## Overview
This project implements an **extractive text summarization system** using **Natural Language Processing (NLP)** techniques.  
The goal is to automatically generate a concise summary by selecting the most important sentences from a given text based on word frequency scoring.

## Key Features
- Text processing using SpaCy
- Tokenization (words and sentences)
- Stopword and punctuation removal
- Word frequency calculation and normalization
- Sentence scoring based on word importance
- Extractive summarization using top-ranked sentences

## Methodology
1. Load and process text using SpaCy English model  
2. Remove stopwords, punctuation, and irrelevant tokens  
3. Compute word frequencies and normalize them  
4. Score each sentence based on word importance  
5. Select top sentences using a compression factor  
6. Generate the final summary by concatenation  

## Techniques Used
- **SpaCy NLP pipeline**
- **Tokenization**
- **Frequency-based sentence scoring**
- **Heap-based ranking (nlargest)**

## Tech Stack
- Python
- SpaCy
- Heapq
- NLP fundamentals

## Output
- Automatically generated extractive summary
- Reduced text while preserving key information

## Use Case
- Document summarization
- Information Retrieval systems
- News and article summarization
- NLP preprocessing for downstream tasks
-------------------------------------------------------------------------------------------------------------------------
# Project 5 (Real-Time Intrusion Detection Pipeline)

## Overview
This project implements a **real-time intrusion detection system (IDS)** using a full **data engineering and machine learning pipeline**.  
It extracts network traffic data, preprocesses it, detects anomalies, and evaluates a classification model in real time.

## Key Features
- Real-time data extraction from API endpoints  
- Timestamp-based feature engineering (Year, Month, Day, Hour, Minute, Second)  
- Data cleaning, outlier detection (Z-score), and standardization  
- Encoding categorical features  
- SVM-based classification with hyperparameter tuning (GridSearchCV)  
- Model evaluation using cross-validation and test split (F1 score & classification report)  
- Continuous monitoring and automated pipeline execution  

## Pipeline Steps
1. **Data Extraction**: Fetch real-time network traffic records from the API  
2. **Data Transformation**:  
   - Convert timestamps to separate features  
   - Encode categorical columns  
   - Detect and remove outliers  
   - Standardize numeric features  
3. **Model Training & Evaluation**:  
   - Train SVM classifier with GridSearchCV  
   - Cross-validation evaluation  
   - F1 score calculation  
   - Prediction on test subset  
4. **Monitoring**: Continuous execution with logging and exception handling  

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn (SVM, preprocessing, cross-validation)  
- Logging for real-time monitoring  
- Requests for API data extraction  

## Output
- Cleaned and standardized dataset  
- Optimized SVM model for intrusion detection  
- Real-time monitoring logs  
- Classification reports and F1 scores  

## Use Case
- Network intrusion detection  
- Real-time anomaly detection systems  
- Cybersecurity and threat monitoring  
----------------------------------------------------------------------------------------------------------------------
# Project 6 (Data Mining)

## Overview
This project performs **data preprocessing and association rule mining** on a community participation dataset. The goal is to uncover **patterns and relationships** between demographic attributes, activities, and social engagement.

The analysis helps understand participant behavior, preferences, and potential correlations between features such as age, gender, hobbies, and social club participation.

## Key Features
- Data cleaning and preprocessing
  - Handling missing values and outliers
  - Encoding categorical variables
  - Feature discretization (e.g., age groups)
- Exploratory Data Analysis (EDA) with visualizations
- Association Rule Mining using **Apriori Algorithm**
  - Extraction of frequent itemsets
  - Generation of actionable rules with support and confidence thresholds
- Output: insights on community participation patterns

## Tech Stack
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Plotly)
- **mlxtend** (TransactionEncoder, Apriori, Association Rules)
- Data preprocessing and normalization (StandardScaler, MinMaxScaler)

## Dataset
- Community participation survey data
- Attributes include:
  - Demographics (Age, Gender, Marital Status)
  - Activity metrics (Time in community, Hobbies, Social clubs, Political/Religious involvement)
  - Purchase behavior (optional)


