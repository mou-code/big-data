"""
Data loading and initialization utilities for the sales prediction project.
"""

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import pyspark.sql.functions as F
from config import SPARK_CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR

class SparkDataLoader:
    def __init__(self):
        """Initialize Spark session with configured settings."""
        self.spark = (SparkSession.builder
                     .appName(SPARK_CONFIG["spark.app.name"])
                     .master(SPARK_CONFIG["spark.master"])
                     .config("spark.executor.memory", SPARK_CONFIG["spark.executor.memory"])
                     .config("spark.driver.memory", SPARK_CONFIG["spark.driver.memory"])
                     .config("spark.sql.shuffle.partitions", SPARK_CONFIG["spark.sql.shuffle.partitions"])
                     .getOrCreate())

    def load_data(self, file_path):
        """
        Load data from various file formats into Spark DataFrame.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pyspark.sql.DataFrame: Loaded data
        """
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension == 'csv':
            return self.spark.read.csv(file_path, header=True, inferSchema=True)
        elif file_extension in ['parquet', 'pq']:
            return self.spark.read.parquet(file_path)
        elif file_extension == 'json':
            return self.spark.read.json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def preprocess_data(self, df):
        """
        Perform initial data preprocessing steps.
        
        Args:
            df (pyspark.sql.DataFrame): Input DataFrame
            
        Returns:
            pyspark.sql.DataFrame: Preprocessed DataFrame
        """
        # Handle missing values
        numeric_columns = [f.name for f in df.schema.fields if f.dataType.typeName() in ['double', 'integer', 'long']]
        string_columns = [f.name for f in df.schema.fields if f.dataType.typeName() == 'string']
        
        # Fill numeric columns with mean
        for col_name in numeric_columns:
            mean_value = df.select(F.mean(col(col_name))).collect()[0][0]
            df = df.fillna(mean_value, subset=[col_name])
        
        # Fill categorical columns with mode
        for col_name in string_columns:
            mode_value = df.groupBy(col_name).count().orderBy('count', ascending=False).first()[0]
            df = df.fillna(mode_value, subset=[col_name])
        
        return df

    def categorize_sales(self, df, sales_column='sales_count_week'):
        """
        Categorize sales into high, medium, and low categories.
        
        Args:
            df (pyspark.sql.DataFrame): Input DataFrame
            sales_column (str): Name of the sales column
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with sales categories
        """
        return df.withColumn('sales_category',
            when(col(sales_column) <= 50, 'low')
            .when((col(sales_column) > 50) & (col(sales_column) <= 200), 'medium')
            .otherwise('high'))

    def save_processed_data(self, df, filename):
        """
        Save processed DataFrame to the processed data directory.
        
        Args:
            df (pyspark.sql.DataFrame): DataFrame to save
            filename (str): Name of the output file
        """
        output_path = str(PROCESSED_DATA_DIR / filename)
        df.write.mode('overwrite').parquet(output_path)

    def close(self):
        """Stop the Spark session."""
        self.spark.stop() 