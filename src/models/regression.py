"""
Regression models implementation using MapReduce paradigm.
"""

from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col
import numpy as np
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

class MapReduceLinearRegression:
    def __init__(self, spark, num_partitions=4):
        """
        Initialize MapReduce Linear Regression.
        
        Args:
            spark: SparkSession object
            num_partitions: Number of partitions for parallel processing
        """
        self.spark = spark
        self.num_partitions = num_partitions
        self.models = []
        self.weights = None
        self.intercept = None
        
    def map_function(self, partition_df):
        """
        Map function to train local models on data partitions.
        
        Args:
            partition_df: DataFrame partition
            
        Returns:
            LinearRegression: Trained model on partition
        """
        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            regParam=0.1,
            elasticNetParam=0.0
        )
        return lr.fit(partition_df)
    
    def reduce_function(self, models):
        """
        Reduce function to combine local models.
        
        Args:
            models: List of trained local models
            
        Returns:
            tuple: (averaged_weights, averaged_intercept)
        """
        weights = np.mean([model.coefficients.toArray() for model in models], axis=0)
        intercept = np.mean([model.intercept for model in models])
        return weights, intercept
    
    def fit(self, df):
        """
        Fit the MapReduce Linear Regression model.
        
        Args:
            df: Input DataFrame with features and label
        """
        # Repartition data
        df_partitioned = df.repartition(self.num_partitions)
        
        # Map phase: Train local models on partitions
        self.models = df_partitioned.mapInPandas(
            lambda partition: [self.map_function(partition)],
            LinearRegression
        ).collect()
        
        # Reduce phase: Combine local models
        self.weights, self.intercept = self.reduce_function(self.models)
        
        return self
    
    def predict(self, df):
        """
        Make predictions using the trained model.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            DataFrame: Original DataFrame with predictions
        """
        if self.weights is None or self.intercept is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Define prediction function
        def predict_udf(features):
            return float(np.dot(features, self.weights) + self.intercept)
        
        # Register UDF
        predict_func = udf(predict_udf, DoubleType())
        
        # Add predictions
        return df.withColumn("prediction", predict_func(col("features")))
    
    def evaluate(self, df):
        """
        Evaluate the model performance.
        
        Args:
            df: Test DataFrame with features and label
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        predictions = self.predict(df)
        
        # Calculate metrics
        mse = predictions.select(
            ((col("prediction") - col("label")) ** 2).alias("mse")
        ).agg({"mse": "avg"}).collect()[0][0]
        
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        total_variance = df.select(
            (col("label") - df.select(col("label").mean()).collect()[0][0]) ** 2
        ).agg({"label": "sum"}).collect()[0][0]
        
        residual_variance = predictions.select(
            (col("prediction") - col("label")) ** 2
        ).agg({"prediction": "sum"}).collect()[0][0]
        
        r2 = 1 - (residual_variance / total_variance)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        } 