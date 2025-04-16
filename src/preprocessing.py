"""
Data preprocessing and feature engineering module.
"""

from pyspark.ml.feature import (VectorAssembler, StandardScaler, StringIndexer, 
                              OneHotEncoder, PCA)
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from config import (NUMERICAL_FEATURES, CATEGORICAL_FEATURES, 
                   BINARY_FEATURES, TARGET_VARIABLE)

class DataPreprocessor:
    def __init__(self, spark):
        """
        Initialize the preprocessor with a Spark session.
        
        Args:
            spark: SparkSession object
        """
        self.spark = spark
        self.fitted_pipeline = None
        
    def create_preprocessing_pipeline(self):
        """
        Create a preprocessing pipeline for feature transformation.
        
        Returns:
            Pipeline: Spark ML Pipeline
        """
        stages = []
        
        # String Indexing for categorical features
        string_indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep")
            for col in CATEGORICAL_FEATURES
        ]
        stages.extend(string_indexers)
        
        # One-Hot Encoding for indexed categorical features
        onehot_encoders = [
            OneHotEncoder(
                inputCol=f"{col}_indexed",
                outputCol=f"{col}_encoded"
            )
            for col in CATEGORICAL_FEATURES
        ]
        stages.extend(onehot_encoders)
        
        # Convert binary features to numeric
        binary_indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_num", handleInvalid="keep")
            for col in BINARY_FEATURES
        ]
        stages.extend(binary_indexers)
        
        # Combine all features into a single vector
        feature_cols = (
            NUMERICAL_FEATURES +
            [f"{col}_encoded" for col in CATEGORICAL_FEATURES] +
            [f"{col}_num" for col in BINARY_FEATURES]
        )
        
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_assembled",
            handleInvalid="keep"
        )
        stages.append(assembler)
        
        # Standardize features
        scaler = StandardScaler(
            inputCol="features_assembled",
            outputCol="features_scaled",
            withStd=True,
            withMean=True
        )
        stages.append(scaler)
        
        # Dimensionality reduction with PCA
        pca = PCA(
            k=10,  # Number of principal components
            inputCol="features_scaled",
            outputCol="features"
        )
        stages.append(pca)
        
        return Pipeline(stages=stages)
    
    def fit_transform(self, df):
        """
        Fit the preprocessing pipeline and transform the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Transformed DataFrame
        """
        pipeline = self.create_preprocessing_pipeline()
        self.fitted_pipeline = pipeline.fit(df)
        return self.fitted_pipeline.transform(df)
    
    def transform(self, df):
        """
        Transform new data using the fitted pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Transformed DataFrame
        """
        if self.fitted_pipeline is None:
            raise ValueError("Pipeline has not been fitted. Call fit_transform first.")
        return self.fitted_pipeline.transform(df)
    
    def prepare_features_for_modeling(self, df, for_classification=False):
        """
        Prepare features for modeling, handling both regression and classification tasks.
        
        Args:
            df: Input DataFrame
            for_classification: Boolean indicating if preparing for classification
            
        Returns:
            DataFrame: DataFrame ready for modeling
        """
        if for_classification:
            # For classification, create categorical target
            df = df.withColumn(
                "label",
                udf(lambda x: 0 if x <= 50 else (1 if x <= 200 else 2), DoubleType())(col(TARGET_VARIABLE))
            )
        else:
            # For regression, use raw sales count
            df = df.withColumn("label", col(TARGET_VARIABLE))
        
        return df.select("features", "label") 