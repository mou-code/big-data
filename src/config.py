"""
Configuration settings for the sales prediction project.
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Spark Configuration
SPARK_CONFIG = {
    "spark.app.name": "Sales_Prediction",
    "spark.master": "local[*]",  # Pseudo-distributed mode
    "spark.executor.memory": "4g",
    "spark.driver.memory": "4g",
    "spark.sql.shuffle.partitions": "8"
}

# Feature Configuration
NUMERICAL_FEATURES = [
    'price', 'primaryPrice', 'stock', 'weight',
    'preparationDays', 'rating_average', 'rating_count'
]

CATEGORICAL_FEATURES = [
    'categoryTitle', 'vendor_name', 'vendor_owner_city',
    'vendor_status_title', 'has_variation', 'has_delivery'
]

BINARY_FEATURES = [
    'isFreeShipping', 'vendor_has_delivery',
    'vendor_freeShippingToIran', 'vendor_freeShippingToSameCity'
]

TARGET_VARIABLE = 'sales_count_week'

# Model Parameters
RANDOM_SEED = 42

# Sales Categories for Classification
SALES_CATEGORIES = {
    'low': (0, 50),
    'medium': (51, 200),
    'high': (201, float('inf'))
}

# Model Training Parameters
TRAIN_TEST_SPLIT_RATIO = 0.8
VALIDATION_SIZE = 0.2  # 20% of training data

# Clustering Parameters
N_CLUSTERS = 5  # Default number of clusters for K-means 