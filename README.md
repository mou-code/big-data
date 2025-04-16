# E-commerce Sales Prediction using PySpark

This project implements a big data analytics solution using PySpark to predict weekly sales count for products on an online shopping website. The implementation includes both regression and classification approaches, along with comprehensive exploratory data analysis (EDA).

## Project Structure

```
sales_prediction_project/
├── data/
│   ├── raw/                # For raw data files
│   └── processed/          # For processed data files
├── notebooks/
│   └── 1_eda.ipynb        # EDA notebook
├── src/
│   ├── config.py          # Configuration settings
│   ├── data_loader.py     # Data loading utilities
│   ├── preprocessing.py    # Data preprocessing
│   ├── models/
│   │   ├── regression.py  # MapReduce Linear Regression
│   │   └── clustering.py  # MapReduce K-Means
│   └── utils.py           # Utility functions
└── requirements.txt        # Project dependencies

```

## Features

1. **EDA Phase**
   - Missing data handling
   - Outlier detection
   - Correlation analysis
   - Dimensionality reduction (PCA)
   - Data visualization

2. **Predictive Analysis**
   - Implementation of various algorithms using MapReduce
   - Model evaluation and comparison
   - Feature importance analysis

3. **Descriptive Analysis**
   - K-Means clustering
   - Association rules mining

## Setup Instructions

1. Install Python 3.8 or higher
2. Install Java 8 or higher (required for Spark)
3. Install Apache Spark
4. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Place your dataset in the data/raw/ directory
7. Run the EDA notebook to understand the data:
   ```bash
    jupyter notebook notebooks/1_eda.ipynb
    ```



## Usage

1. Place your dataset in the `data/raw/` directory
2. Run the notebooks in order:
   - Start with `1_eda.ipynb` for exploratory analysis
   - Proceed with `2_preprocessing.ipynb` for data preparation
   - Finally, run `3_modeling.ipynb` for model training and evaluation

## Key Insights Analyzed

1. Sales, stock, and price relationships
2. Impact of free shipping on sales
3. Category-wise performance analysis
4. Price discount effects
5. Vendor performance analysis
6. Shipping and delivery impact
7. Product characteristics influence

## Models Implemented

1. Regression Models:
   - Linear Regression (MapReduce)
   - Random Forest Regression

2. Classification Models:
   - Logistic Regression (MapReduce)
   - XGBoost

3. Clustering:
   - K-Means clustering

## Requirements

See `requirements.txt` for detailed dependencies. 
