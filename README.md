# E-Commerce Product Sales Prediction

This project aims to predict product performance on online shopping platforms using machine learning models and data analytics. By analyzing factors like price, ratings, and weekly sales, we provide accurate forecasts to help vendors better manage inventory and marketing planning.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Pipeline](#project-pipeline)
- [Key Insights](#key-insights)
- [Models and Results](#models-and-results)
- [Installation and Setup](#installation-and-setup)
- [Running on AWS EMR](#running-on-aws-emr)
- [Future Work](#future-work)
- [Team Members](#team-members)

## 🔍 Project Overview

This big data project analyzes the [BaSalam e-commerce dataset](https://www.kaggle.com/datasets/radeai/basalam-comments-and-products/data) (containing over 2.4 million product records) to predict product performance scores. Using **Apache Spark (PySpark)** for distributed computing, the system processes and analyzes this large-scale dataset to help vendors understand which factors contribute most to product success.

Our main objective is to develop accurate performance forecasting models considering price, ratings, delivery options, and weekly sales patterns. These insights enable vendors to optimize inventory management, pricing strategies, and marketing planning.

**Key Project Goals:**
- Process and analyze a large e-commerce dataset using distributed computing
- Identify the most significant factors influencing product performance
- Develop predictive models to forecast product success metrics
- Extract actionable business insights for vendors and marketplace operators
- Implement association rule mining to discover hidden patterns in product performance

By leveraging PySpark's distributed processing capabilities, we've successfully handled this big data challenge and produced valuable insights that would be impractical to obtain using traditional data analysis methods.

## 📊 Dataset Description

The BaSalam dataset is a large-scale e-commerce dataset containing over 2.4 million product records from the BaSalam online marketplace. This big data challenge necessitated the use of **PySpark** for efficient distributed processing. The dataset includes:

- Product identifiers and names (`_id`, `name`)
- Pricing information (`price`, `primaryPrice`)
- Inventory and stock levels (`stock`)
- Performance metrics (`_score`, `sales_count_week`, `rating_average`, `rating_count`)
- Vendor details (`vendor_name`, `vendor_id`, `vendor_score`)
- Shipping and delivery options (`has_delivery`, `isFreeShipping`, `vendor_freeShippingToIran`)
- Product categorization (`categoryTitle`, `categoryId`)
- Product status information (`status_id`, `status_title`, `IsAvailable`, `IsSaleable`)
- Product characteristics (`weight`, `preparationDays`, `has_variation`)
- Media content availability (`photo_MEDIUM`, `photo_SMALL`, `video_ORIGINAL`)

The dataset's size and complexity required specialized big data techniques for analysis and modeling, making it an excellent case study for applied big data analytics in e-commerce.

## 🔄 Project Pipeline

![pipeline](https://github.com/user-attachments/assets/66fe35ec-2d4b-4667-82d8-fdd9888be5e4)

The project follows a structured big data processing and machine learning approach. Due to the large size of the BaSalam dataset (over 2.4 million records), we leveraged **Apache Spark (PySpark)** for distributed computing to efficiently process and analyze the data.

### 1. Data Study
- Explored data types (categorical, numerical, binary, boolean)
- Identified and analyzed null values across all columns
- Generated histograms and distribution plots for numerical columns
- Examined cardinality of categorical features

### 2. Data Preprocessing
- Dropped columns with high null percentages (>80%)
- Split data into training and test sets
- Removed irrelevant columns (IDs, photo URLs, etc.)
- For association rules: dropped rows with nulls (6.19% of data)
- For predictive models: imputed nulls with median (numerical) and mode (categorical)
- Dropped high-cardinality categorical columns
- Applied one-hot encoding to remaining categorical columns
- Handled multicollinearity by dropping highly correlated features
- Selected features based on correlation with target variable

### 3. Data Visualization & Exploration
- Generated correlation heatmaps to identify relationships
- Visualized categorical relationships using bar charts

### 4. Insight Extraction
- Performed statistical analysis to extract business insights
- Conducted comparative analysis across product categories
- Identified key factors contributing to high product scores

### 5. Model Training
- Implemented multiple regression models using PySpark ML
- Trained classification models for segmentation
- Applied MapReduce KNN for distributed neighbor-based classification
- Created association rules models to discover patterns
- Performed hyperparameter tuning for SVM model
- Utilized Spark's distributed computing for efficient model training

### 6. Evaluation
- Calculated RMSE and R² for regression models
- Generated confusion matrices for classification models
- Performed cross-validation to ensure model robustness
- Identified the most important features for prediction
- Compared model performance across different algorithms

## 💡 Key Insights & Data Visualization

Our big data analysis revealed several important insights about product performance through extensive visualization techniques:

### 1. Price & Discounts Impact

**Key Finding**: We found a negative correlation (-0.12) between price and score, indicating that cheaper products tend to receive higher scores. Our visualization analysis showed:

- Discount analysis revealed an optimal "sweet spot":
  - Products with discounts between 70-80% perform best (highest average scores)
  - Extreme discounts (90-100%) trigger customer skepticism ("too good to be true")
  - Small discounts (0-10%) show minimal impact on performance scores
![discount](https://github.com/user-attachments/assets/027bf19f-e38a-4ee3-bf1a-b966fd159c6b)

### 2. Product Variations Matter

Products with variations (different sizes/colors) significantly outperform those without:

| Has Variation | Average Score |
|--------------|---------------|
| False        | 68.17         |
| True         | 90.20         |

Our analysis shows that offering product variations increases the average score by 32.3%, suggesting that customer choice is highly valued.

### 3. Visual Content Drives Performance

Visual representation dramatically impacts product performance:

| Has Photo | Has Video | Average Product Score |
|-----------|-----------|----------------------|
| False     | False     | 18.22                | 
| False     | True      | 13.89                | 
| True      | False     | 68.59                |
| True      | True      | 89.28                |

Products with both photos and videos show a score increase compared to those without any visual content, demonstrating the critical importance of multimedia in e-commerce.

### 4. Rating Influence

Our visualization analysis of ratings revealed:

- Most products with ratings cluster between 4-5 stars
- A large percentage (~40%) of products have no ratings (rating_average = 0)
- Strong positive correlation between rating metrics and performance score
- Products with ratings above 4.5 have significantly higher average scores
- Rating count (number of reviews) is as important as rating average
![rating](https://github.com/user-attachments/assets/f4e6a79a-ff7d-4ab3-b08b-0130d7772434)

### 5. Free Shipping Economics

![Free Shipping vs Price](/api/placeholder/600/400)

Our analysis of shipping options revealed interesting pricing strategies:

| Free Shipping | Average Price | 
|---------------|---------------|
| False         | 2719.07      |
| True          | 7034.42      |

Free shipping products are more expensive than non-free shipping ones, suggesting vendors offset shipping costs through higher product pricing.

### 6. Category Performance Analysis
![category](https://github.com/user-attachments/assets/1c789140-6834-4890-8804-62a4c96f9e96)

## 📈 Models and Results

We implemented several big data machine learning models using PySpark's distributed computing capabilities to predict product performance scores:

### Regression Models

| Model                     | Train RMSE | Train R² | Test RMSE | Test R² |
|---------------------------|------------|----------|-----------|---------|
| Linear Regression         | 54.39      | 0.666    | 54.53     | 0.664   |
| Random Forest Regressor   | 46.61      | 0.754    | 46.55     | 0.756   |
| Decision Tree Regressor   | 41.52      | 0.805    | 41.37     | 0.807   |
| Generalized Linear Reg    | 54.39      | 0.666    | 54.53     | 0.665   |
| GBT Regressor             | 40.90      | 0.811    | 40.76     | 0.813   |
| Isotonic Regression       | 79.71      | 0.283    | 79.76     | 0.283   |

The **Gradient Boosted Tree (GBT) Regressor** achieved the best performance with a test R² of 0.813 and RMSE of 40.76, indicating that it captured approximately 81.3% of the variance in product performance scores.

### Classification Models

We also implemented an **SVM model** with hyperparameter tuning:

- Conducted grid search over combinations of:
  - MaxIter: [10, 50, 100]
  - RegParam: [0.01, 0.1, 1.0]
- Best parameters: MaxIter=10, RegParam=0.01
- Performance metrics:
  - Training Accuracy: 86.73%
  - Validation Accuracy: 86.75%
  - Test Accuracy: 86.84%

### Custom MapReduce KNN Implementation

We developed a distributed K-Nearest Neighbors algorithm using MapReduce principles:

1. **Data Preparation**:
   - Categorized score into 3 classes (0-150, 150-300, >300)
   - Split data into training and testing sets
   
2. **MapReduce Implementation**:
   - **Map Phase**: Computed cosine similarity between test point and each training point within partitions
   - **Reduce Phase**: Aggregated neighbors across partitions and selected top-k global neighbors
   - **Prediction**: Used weighted voting based on similarity scores and inverse class frequency

3. **Evaluation**:
   - Generated confusion matrix to assess classification performance
   - Applied to test samples for prediction validation

### Association Rule Mining

We implemented association rule mining to discover hidden patterns:

1. **Data Transformation**:
   - Converted numerical columns to categorical
   - Created itemsets by concatenating column values
   
2. **Rule Discovery**:
   - Applied FP-Growth algorithm on PySpark
   - Filtered for minimum support and confidence thresholds
   
3. **Key Findings**:
   - Products with no ratings tend to have low engagement metrics
   - Extremely low rating counts consistently link to low sales
   - Low-performing products share a profile of low stock and low weekly sales
   - Well-stocked products tend to be high-priced without free shipping or variations

The diversity of our modeling approaches allowed us to both predict performance scores (regression models) and discover meaningful patterns (association rules) in this large e-commerce dataset.

## ☁️ Running on AWS EMR

Due to the large dataset size (2.4M+ records), we deployed this project on Amazon EMR for distributed computing to handle the big data processing requirements:

### Deployment Process

1. **Data Preparation**:
   - Uploaded the BaSalam dataset and notebook files to Amazon S3 bucket (`s3://bigdata-3400/BaSalam.products.csv`)
   - Prepared preprocessing and analysis code for PySpark execution

2. **EMR Cluster Creation**:
   - Created an Amazon EMR Cluster with the following configurations:
     - **Application Bundle**:
       - Hadoop 3.4.1 (for distributed file system)
       - JupyterHub 1.5 (for interactive notebook interface)
       - Spark 3.5.4 (for big data processing)
     
     - **Instance Configuration**:
       - 1 Primary node: m4.large (for cluster management)
       - 1 Core node: m4.large (for data processing and storage)
       - 1 Task node: m4.large (for additional processing power)

3. **Security Configuration**:
   - Added port 9443 to the security group to allow JupyterHub access
   - Set up appropriate IAM roles for S3 data access

4. **Cluster Connection**:
   - Connected to JupyterHub via `https://<master-node-public-dns>:9443`
   - Cluster DNS: `ec2-18-216-234-213.us-east-2.compute.amazonaws.com`
   - Used default credentials (username: jovyan, password: jupyter)

5. **Distributed Processing Setup**:
   - Created new Jupyter Notebook with PySpark kernel for fully distributed mode
   - Configured notebook to read data directly from S3
   - Set up Spark session with appropriate memory and executor configurations

6. **Execution**:
   - Performed data preprocessing in distributed mode across the cluster
   - Executed modeling and analysis tasks using Spark's distributed computing capabilities
   - Successfully processed the entire 2.4M+ record dataset that would have been impractical on a single machine
![AWS](https://github.com/user-attachments/assets/351b6a29-d335-4c6c-9891-5e39ae403f0a)

## 🚀 Future Work & Enhancements

Based on our project experience and findings, we've identified several promising directions for future work:

### Data Processing Enhancements
- **Advanced Categorical Handling**: Rather than dropping columns with high cardinality, explore clustering techniques to group similar values or consolidate rare values under an "Other" category
- **Feature Engineering**: Create additional derived features capturing interactions between price, rating, and delivery options
- **Dimensionality Reduction**: Apply PCA or t-SNE to visualize high-dimensional relationships in the data

### Model Improvements
- **Deep Learning Models**: Implement neural network architectures specialized for tabular data
- **Ensemble Techniques**: Create stacked or blended models combining the strengths of multiple algorithms
- **AutoML Integration**: Implement automated hyperparameter tuning using Spark's hyperopt or similar distributed optimization frameworks

### System Implementation
- **Real-time Prediction API**: Build an API service that vendors can query for real-time performance predictions
- **Vendor Dashboard**: Create an interactive dashboard with performance insights and recommendations

## Contributors <a name = "Contributors"></a>

<table>
  <tr>
    <td align="center">
    <a href="https://github.com/Menna-Ahmed7" target="_black">
    <img src="https://avatars.githubusercontent.com/u/110634473?v=4" width="150px;" alt="https://github.com/Menna-Ahmed7"/>
    <br />
    <sub><b>Mennatallah Ahmed</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MostafaBinHani" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119853216?v=4" width="150px;" alt="https://github.com/MostafaBinHani"/>
    <br />
    <sub><b>Mostafa Hani</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MohammadAlomar8" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119791309?v=4" width="150px;" alt="https://github.com/MohammadAlomar8"/>
    <br />
    <sub><b>Mohammed Alomar</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/mou-code" target="_black">
    <img src="https://avatars.githubusercontent.com/u/123744354?v=4" width="150px;" alt="https://github.com/mou-code"/>
    <br />
    <sub><b>Moustafa Mohammed</b></sub></a>
    </td>
  </tr>
 </table>
