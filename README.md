# House-Price-Prediction
This project aims to predict house prices using machine learning models, based on data provided by the Kaggle House Prices competition. By analyzing various property features, the project seeks to develop an accurate prediction model that can be applied in real estate pricing and investment analysis.

## *Table of Contents*
- [Project Overview](#project-overview)
- [Data Overview](#data-overview)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Selection and Training](#model-selection-and-training)
- [Model Evaluation and Results](#model-evaluation-and-results)
- [Conclusion](#conclusion)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [License](#license)

---

## *Project Overview*
The objective of this project is to develop a machine learning model capable of accurately predicting housing prices. Utilizing various regression algorithms, the model identifies and leverages key house features for high-accuracy price predictions. This project is useful for real estate agents, investors, and data science practitioners interested in predictive modeling for the housing market.

## *Data Overview*
The dataset consists of 81 features describing various attributes of houses, such as lot size, building type, quality, condition, different area measurements, and the sale price. Key features include:

- *LotArea*: The lot size in square feet.
- *YearBuilt*: The year the house was built.
- *OverallQual*: An overall rating of the house quality.
- *GrLivArea*: The above-ground living area in square feet.
- *SalePrice*: The target variable representing the sale price of the house.

These features provide an extensive view of the factors that contribute to house pricing, allowing the model to capture intricate relationships.

## *Data Preprocessing*
To prepare the data for machine learning, the following preprocessing steps were conducted:

1. *Handling Missing Values*: Missing values were filled based on feature types—numerical features with their means, and categorical features with the mode or a placeholder value.
2. *Encoding Categorical Data*: Label encoding was applied to categorical columns, converting them to numeric representations suitable for model training.
3. *Scaling*: Numerical features were scaled using MinMaxScaler to normalize data ranges and reduce bias from scale differences.
4. *Unimportant Feature Removal*: Columns with high percentages of missing values or low variance (such as Alley, PoolQC, Fence) were removed to improve model efficiency.

## *Feature Engineering*
Additional features were created to enhance the model's predictive power:

- *Age*: Calculated as the difference between YearBuilt and YrSold, representing the age of the house at the time of sale.
- *TotalArea*: Combined areas from features like GrLivArea, GarageArea, and TotalBsmtSF to reflect the total usable space, which has a strong influence on price.
- *Interaction Terms*: Created new interaction features, capturing relationships between other predictive features, improving model understanding.

## *Model Selection and Training*
Four models were evaluated for performance on the regression task:

1. *Linear Regression*: Used as a baseline model to identify general trends. Limited by its linear assumptions.
2. *Decision Tree Regressor*: Captures non-linear relationships but is prone to overfitting on deep trees.
3. *Random Forest Regressor*: An ensemble method that aggregates multiple decision trees, reducing overfitting and increasing accuracy.
4. *LightGBM Regressor*: A gradient boosting model optimized for performance, particularly effective for high-dimensional datasets.

Each model was trained and hyperparameters tuned to maximize accuracy. The final model, *LightGBM*, was selected due to its high accuracy and speed in handling complex feature relationships.

## *Model Evaluation and Results*
The model performance was evaluated using the following metrics:

- *Mean Squared Error (MSE)*: Measures the average squared difference between actual and predicted prices. A lower MSE indicates better model performance.
- *R² Score*: Reflects the proportion of variance in sale price explained by the model, with a score close to 1 indicating high accuracy.

| Model                    | MSE          | R² Score   |
|--------------------------|--------------|------------|
| Linear Regression        | 17,200,000   | 0.85       |
| Decision Tree Regressor  | 10,500,000   | 0.90       |
| Random Forest Regressor  | 7,600,000    | 0.95       |
| *LightGBM Regressor*   | *5,316,154* | *0.999*  |

The *LightGBM model* achieved the best results, with an MSE of approximately 5.3 million and an R² score of 0.999, making it the final model used for Kaggle submission.

## *Conclusion*
The House Price Prediction project demonstrates the effectiveness of machine learning models in predicting real estate prices. Through comprehensive data preprocessing, feature engineering, and model tuning, the *LightGBM model* provided the highest accuracy. This model is suitable for applications in real estate market analysis and investment planning.

## *Installation and Setup*

1. Clone the repository to your local machine.
   bash
   git clone https://github.com/yourusername/house-price-prediction.git
# Installation and Setup

markdown
# Install required dependencies:
bash
pip install -r requirements.txt
# Download the dataset from the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) and place it in the `data/` directory.

# Run the Jupyter Notebook:
bash
jupyter notebook

# Usage

```markdown
To run the model and evaluate performance:
1. Preprocess the data by running the cells in house-prediction.ipynb up to the training stage.
2. Train the model and generate predictions on the test set.
3. Submit predictions to Kaggle by generating a CSV file using the code in the notebook.
