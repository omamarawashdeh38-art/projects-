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

