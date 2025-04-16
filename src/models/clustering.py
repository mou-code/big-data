"""
Clustering models implementation for descriptive analysis.
"""

from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col
import numpy as np
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from config import N_CLUSTERS

class MapReduceKMeans:
    def __init__(self, spark, k=N_CLUSTERS, num_partitions=4):
        """
        Initialize MapReduce K-Means clustering.
        
        Args:
            spark: SparkSession object
            k: Number of clusters
            num_partitions: Number of partitions for parallel processing
        """
        self.spark = spark
        self.k = k
        self.num_partitions = num_partitions
        self.local_models = []
        self.global_centroids = None
        
    def map_function(self, partition_df):
        """
        Map function to train local K-Means models on data partitions.
        
        Args:
            partition_df: DataFrame partition
            
        Returns:
            KMeans: Trained model on partition
        """
        kmeans = KMeans(
            k=self.k,
            featuresCol="features",
            predictionCol="cluster",
            maxIter=20,
            seed=42
        )
        return kmeans.fit(partition_df)
    
    def reduce_function(self, models):
        """
        Reduce function to combine local centroids into global centroids.
        
        Args:
            models: List of trained local models
            
        Returns:
            ndarray: Global centroids
        """
        all_centroids = np.vstack([model.clusterCenters() for model in models])
        
        # Use K-Means to find global centroids from local centroids
        kmeans = KMeans(
            k=self.k,
            featuresCol="features",
            predictionCol="cluster",
            maxIter=50,
            seed=42
        )
        
        # Convert centroids to DataFrame
        centroids_df = self.spark.createDataFrame(
            [tuple([c]) for c in all_centroids],
            ["features"]
        )
        
        global_model = kmeans.fit(centroids_df)
        return global_model.clusterCenters()
    
    def fit(self, df):
        """
        Fit the MapReduce K-Means clustering model.
        
        Args:
            df: Input DataFrame with features
        """
        # Repartition data
        df_partitioned = df.repartition(self.num_partitions)
        
        # Map phase: Train local models on partitions
        self.local_models = df_partitioned.mapInPandas(
            lambda partition: [self.map_function(partition)],
            KMeans
        ).collect()
        
        # Reduce phase: Combine local models
        self.global_centroids = self.reduce_function(self.local_models)
        
        return self
    
    def predict(self, df):
        """
        Assign clusters to data points using global centroids.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            DataFrame: Original DataFrame with cluster assignments
        """
        if self.global_centroids is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Define prediction function
        def predict_udf(features):
            distances = [np.linalg.norm(features - centroid) for centroid in self.global_centroids]
            return int(np.argmin(distances))
        
        # Register UDF
        cluster_assign = udf(predict_udf, IntegerType())
        
        # Add cluster assignments
        return df.withColumn("cluster", cluster_assign(col("features")))
    
    def analyze_clusters(self, df):
        """
        Analyze the characteristics of each cluster.
        
        Args:
            df: DataFrame with cluster assignments and original features
            
        Returns:
            dict: Dictionary containing cluster analysis results
        """
        predictions = self.predict(df)
        
        # Calculate cluster sizes
        cluster_sizes = predictions.groupBy("cluster").count().collect()
        
        # Calculate cluster statistics
        cluster_stats = predictions.groupBy("cluster").agg({
            "sales_count_week": "mean",
            "price": "mean",
            "rating_average": "mean"
        }).collect()
        
        # Prepare analysis results
        analysis = {
            "cluster_sizes": {row["cluster"]: row["count"] for row in cluster_sizes},
            "cluster_stats": {
                row["cluster"]: {
                    "avg_sales": row["avg(sales_count_week)"],
                    "avg_price": row["avg(price)"],
                    "avg_rating": row["avg(rating_average)"]
                }
                for row in cluster_stats
            }
        }
        
        return analysis 