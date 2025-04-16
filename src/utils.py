"""
Utility functions for the sales prediction project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, count, mean, stddev
import numpy as np

def plot_missing_values(df):
    """
    Plot missing values analysis.
    
    Args:
        df: Spark DataFrame
    """
    total_count = df.count()
    missing_counts = []
    
    for column in df.columns:
        missing_count = df.filter(col(column).isNull()).count()
        missing_percentage = (missing_count / total_count) * 100
        missing_counts.append({
            'column': column,
            'missing_percentage': missing_percentage
        })
    
    missing_df = pd.DataFrame(missing_counts)
    missing_df = missing_df.sort_values('missing_percentage', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(missing_df['column'], missing_df['missing_percentage'])
    plt.xlabel('Missing Percentage')
    plt.title('Missing Values Analysis')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, columns):
    """
    Plot correlation matrix for specified columns.
    
    Args:
        df: Spark DataFrame
        columns: List of column names to include in correlation matrix
    """
    correlation_data = df.select(columns).toPandas()
    correlation_matrix = correlation_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

def detect_outliers(df, column, threshold=3):
    """
    Detect outliers using z-score method.
    
    Args:
        df: Spark DataFrame
        column: Column name to check for outliers
        threshold: Z-score threshold for outlier detection
        
    Returns:
        DataFrame: DataFrame with outlier indicators
    """
    stats = df.select(
        mean(col(column)).alias('mean'),
        stddev(col(column)).alias('stddev')
    ).collect()[0]
    
    mean_val = stats['mean']
    stddev_val = stats['stddev']
    
    return df.withColumn(
        f'{column}_zscore',
        (col(column) - mean_val) / stddev_val
    ).withColumn(
        f'{column}_is_outlier',
        (col(f'{column}_zscore').abs() > threshold)
    )

def evaluate_regression_model(predictions, label_col="label", prediction_col="prediction"):
    """
    Evaluate regression model performance.
    
    Args:
        predictions: DataFrame with actual and predicted values
        label_col: Name of the label column
        prediction_col: Name of the prediction column
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Calculate MSE
    mse = predictions.select(
        ((col(prediction_col) - col(label_col)) ** 2).alias("squared_error")
    ).agg({"squared_error": "avg"}).collect()[0][0]
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    
    # Calculate R-squared
    total_variance = predictions.select(
        (col(label_col) - predictions.select(mean(label_col)).collect()[0][0]) ** 2
    ).agg({"label": "sum"}).collect()[0][0]
    
    residual_variance = predictions.select(
        (col(prediction_col) - col(label_col)) ** 2
    ).agg({prediction_col: "sum"}).collect()[0][0]
    
    r2 = 1 - (residual_variance / total_variance)
    
    # Calculate MAE
    mae = predictions.select(
        (abs(col(prediction_col) - col(label_col))).alias("abs_error")
    ).agg({"abs_error": "avg"}).collect()[0][0]
    
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }

def plot_actual_vs_predicted(predictions, label_col="label", prediction_col="prediction"):
    """
    Plot actual vs predicted values.
    
    Args:
        predictions: DataFrame with actual and predicted values
        label_col: Name of the label column
        prediction_col: Name of the prediction column
    """
    pred_data = predictions.select(label_col, prediction_col).toPandas()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_data[label_col], pred_data[prediction_col], alpha=0.5)
    plt.plot([pred_data[label_col].min(), pred_data[label_col].max()],
             [pred_data[label_col].min(), pred_data[label_col].max()],
             'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.show()

def save_model_metrics(metrics, model_name, output_path):
    """
    Save model evaluation metrics to a file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model
        output_path: Path to save the metrics
    """
    with open(output_path, 'a') as f:
        f.write(f"\n{model_name} Performance Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("-" * 50 + "\n") 