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
  ---------------------------------------------------------------------------------------------------------------------------------------
  # Project 7 (3D Human Pose Projection using Deep Learning Concepts)

## Overview
This project demonstrates **3D human pose representation** and its **perspective projection** onto 2D planes from multiple camera angles. The system simulates different viewpoints and visualizes the projected poses for analysis and model development.

## Key Features
- 3D points representation of human body joints  
- Definition of body parts: left/right arms, legs, torso, and head  
- Perspective projection from multiple camera positions  
- Visualization of projected poses in 2D  
- Configurable camera distance and projection plane  

## Tech Stack
- Python  
- NumPy  
- Matplotlib  

## How It Works
1. Define 3D coordinates of human pose points  
2. Group points into body parts for visualization  
3. Define multiple camera positions around the pose  
4. Apply **perspective projection** to map 3D points to 2D plane  
5. Visualize projected poses using Matplotlib  

## Visualization
- Plots show the human pose from 8 different camera angles (0° to 315°)  
- Each plot connects the joints of body parts for a clear 2D representation
--------------------------------------------------------------------------------------------------------------------------
# 3D Data Cube Operations for Deep Learning

## Overview
This project demonstrates **3D data cube operations** applied to sales or inventory data across **locations, time (quarters), and item types**. The focus is on performing **OLAP-style operations** such as roll-up, drill-down, slicing, dicing, and pivoting, which are commonly used in **data analysis and preprocessing for deep learning models**.

## Key Features
- Creation of a 3D data cube (Location × Time × Item Types)  
- **Roll-up**: Aggregate data across time (quarters)  
- **Drill-down**: Extract detailed data for specific quarters  
- **Slicing**: Filter data for a specific location  
- **Dicing**: Filter data for subcubes (specific locations and quarters)  
- **Pivoting**: Change the orientation/view of the data cube for analysis  

## Tech Stack
- Python  
- NumPy  
- Pandas  

## Data Structure
- **Locations**: Chennai, Kolkata, Mumbai, Delhi  
- **Time**: Q1, Q2, Q3, Q4  
- **Item Types**: Type A, Type B  

- Data cube shape: `(Quarters × Locations × Item Types)` → `(4 × 4 × 2)`  

## How It Works
1. Define a 3D NumPy array representing the data cube  
2. Implement functions for common cube operations:  
   - `roll_up()`: Aggregate values across time  
   - `drill_down()`: Extract specific quarter details  
   - `slice_data()`: Filter by location  
   - `dice_data()`: Filter by a subcube  
   - `pivot_data()`: Swap axes for alternative views  
3. Display results for each operation  
--------------------------------------------------------------------------------------------------------------------------------
# Deep Learning Project 3: Manual Neural Network for Moon Dataset

## Overview
This project demonstrates the **implementation of a neural network from scratch** to classify points in the **Moon Dataset**, which is a simple 3D dataset for binary classification.  
The purpose of this project is to **understand the fundamental mechanics of deep learning**, including forward propagation, backward propagation, gradient computation, and weight updates using **Stochastic Gradient Descent (SGD)** without relying on high-level libraries like TensorFlow or PyTorch.

By manually implementing the neural network, this project allows you to:
- Gain a deep understanding of **how neural networks learn**
- Visualize **loss reduction over training**
- Explore **basic activation functions, loss functions, and optimization methods**

## Dataset
- **Filename:** `moonDataset.csv`
- **Features:** 3 input variables (`x1, x2, x3`) representing the coordinates of each point
- **Target:** Binary classification (0 or 1)
- **Dataset size:** Variable (depending on CSV)
- The dataset is ideal for **educational purposes** and simple neural network experiments.

## Neural Network Architecture
- **Input Layer:** 3 neurons corresponding to input features
- **Hidden Layer:** 2 neurons  
  - Each neuron uses a **sigmoid activation function**
- **Output Layer:** 1 neuron (sigmoid activation) to predict the probability of class 1
- **Weights and Biases:** Initialized manually with small values
- **Loss Function:** Squared error  
  \( Loss = (Target - Output)^2 \)
- **Optimization:** Stochastic Gradient Descent (SGD)
- **Number of Epochs:** 20 (adjustable)
- **Learning Rate:** 0.1 (adjustable)

## Implementation Steps

### 1. Data Loading
- Load the dataset using **Pandas**
- Separate **features** (`X`) and **labels** (`y`)

### 2. Initialization
- Initialize weights (`W1`, `W2`, `W3`) and biases (`b1`, `b2`, `b3`) manually
- Set learning rate and number of epochs

### 3. Forward Propagation
- Compute **hidden layer outputs** using:  
  \( h = \sigma(W \cdot X + b) \)
- Compute **output layer value** using:  
  \( y_{pred} = \sigma(W_{out} \cdot h + b_{out}) \)

### 4. Loss Calculation
- Use **squared error**:  
  \( L = (y_{true} - y_{pred})^2 \)
- Track **average loss per epoch**

### 5. Backward Propagation
- Compute **gradients of loss w.r.t output**
- Propagate gradients to **hidden layer**
- Update **weights and biases** using SGD:  
  \( W = W - \eta \cdot \text{gradient} \)

### 6. Training Loop
- Iterate over all epochs
- Iterate over all samples (online learning / SGD)
- Update weights and biases after each sample
- Track **loss history** for plotting

### 7. Visualization
- Plot **loss curve** over epochs using Matplotlib
- Allows observation of **network convergence**

## Key Features
- **Manual neural network implementation**: No TensorFlow/PyTorch
- **Educational focus**: Understand each step of training
- **Loss tracking**: See how network improves over time
- **Flexible**: Can add neurons, layers, or change learning rate
- **Visualization**: Loss curve helps interpret training progress
--------------------------------------------------------------------------------------------------------------------------
