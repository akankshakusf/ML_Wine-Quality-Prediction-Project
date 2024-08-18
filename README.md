# ML_Wine-Quality-Prediction-Project
This is my ML Project for USF

# Objective

This project aims to predict the quality of wine using data related to red and white variants of the Portuguese "Vinho Verde" wine. The project involves data preprocessing, feature engineering, model selection, hyperparameter tuning, and model evaluation.

## Data Description

The dataset used in this project includes various attributes related to the quality of wine. The dataset contains both red and white wine variants, and the columns in the dataset are as follows:

- **fixed acidity**: Fixed acidity levels in the wine.
- **volatile acidity**: Volatile acidity levels, indicating vinegar taste.
- **citric acid**: Citric acid levels contributing to the freshness of wine.
- **residual sugar**: Amount of sugar remaining after fermentation.
- **chlorides**: Chloride content in wine.
- **free sulfur dioxide**: Free form of SO2 in wine.
- **total sulfur dioxide**: Bound and free forms of SO2 in wine.
- **density**: Density of wine.
- **pH**: Acidity level of wine.
- **sulphates**: Sulfate content, acts as an antimicrobial agent.
- **alcohol**: Alcohol content in wine.
- **quality**: Quality score of wine (target variable, originally on a scale from 0 to 10).
- **type**: Type of wine (red or white).

## Methodology

The methodology for this project involves several key steps:

1. **Data Preprocessing**:
   - **Histogram and Statistical Analysis**: Assessed the distribution of the `quality` variable.
   - **Boxplots**: Visualized the relationship between wine features and quality across different wine types.
   - **Correlation Analysis**: Created correlation heatmaps for both red and white wines to identify relationships between features.
   - **Binning Quality**: Binned the `quality` variable into binary categories (high quality: 1, low quality: 0) to simplify prediction.

2. **Model Development**:
   - **Logistic Regression**: Conducted grid search for hyperparameter tuning, evaluated the model using accuracy, cross-validation score, and ROC-AUC score.
   - **RandomForestClassifier**: Implemented hyperparameter tuning using grid search, evaluated the model's performance, and plotted the ROC curve.
   - **KNeighborsClassifier**: Performed grid search for optimal hyperparameters, and evaluated the model using accuracy, cross-validation score, and ROC-AUC score.

3. **Model Evaluation**:
   - Evaluated models using metrics such as accuracy, cross-validation score, ROC-AUC score, and classification reports.
   - Plotted ROC curves for visual comparison of model performance.
   - Generated confusion matrices to understand the model predictions better.

## Key Features and Insights

### **Feature Engineering**:
- **Correlation Analysis**: Identified and analyzed correlations between wine features.
- **Quality Binning**: Simplified the target variable by binning quality scores into binary categories.

### **Model Performance Summary**

- **Logistic Regression**:
  - **Best Parameters (Red Wine)**: `solver='liblinear', C=100, penalty='l1'`
  - **Red Wine Accuracy**: [Provide the accuracy score here]
  - **Red Wine Cross Validation Score**: [Provide the cross-validation score here]
  - **Red Wine ROC-AUC Score**: [Provide the ROC-AUC score here]

  - **Best Parameters (White Wine)**: `solver='liblinear', C=100, penalty='l1'`
  - **White Wine Accuracy**: [Provide the accuracy score here]
  - **White Wine Cross Validation Score**: [Provide the cross-validation score here]
  - **White Wine ROC-AUC Score**: [Provide the ROC-AUC score here]

- **RandomForestClassifier**:
  - **Best Parameters (Red Wine)**: `n_estimators=48, max_features='sqrt', max_depth=4, min_samples_split=5, bootstrap=False`
  - **Red Wine Accuracy**: [Provide the accuracy score here]
  - **Red Wine Cross Validation Score**: [Provide the cross-validation score here]
  - **Red Wine ROC-AUC Score**: [Provide the ROC-AUC score here]

  - **Best Parameters (White Wine)**: `n_estimators=48, max_features='sqrt', max_depth=4, min_samples_split=5, bootstrap=False`
  - **White Wine Accuracy**: [Provide the accuracy score here]
  - **White Wine Cross Validation Score**: [Provide the cross-validation score here]
  - **White Wine ROC-AUC Score**: [Provide the ROC-AUC score here]

- **KNeighborsClassifier**:
  - **Best Parameters (Red Wine)**: `n_neighbors=100, weights='distance', algorithm='ball_tree'`
  - **Red Wine Accuracy**: [Provide the accuracy score here]
  - **Red Wine Cross Validation Score**: [Provide the cross-validation score here]
  - **Red Wine ROC-AUC Score**: [Provide the ROC-AUC score here]

  - **Best Parameters (White Wine)**: `n_neighbors=100, weights='distance', algorithm='ball_tree'`
  - **White Wine Accuracy**: [Provide the accuracy score here]
  - **White Wine Cross Validation Score**: [Provide the cross-validation score here]
  - **White Wine ROC-AUC Score**: [Provide the ROC-AUC score here]

### **Comparison to Baseline**

The **Logistic Regression** model emerged as the best model for both red and white wines. It provided the highest ROC-AUC scores, making it the most effective at predicting wine quality for this dataset.

## Research Questions and Answers

**Q1. What is the best model for predicting wine quality from the provided dataset?**

- **Answer**: The **Logistic Regression** model with `solver='liblinear', C=100, penalty='l1'` was identified as the best model for predicting wine quality for both red and white wine variants. It outperformed other models in terms of ROC-AUC score and accuracy.

**Q2. How do different machine learning models compare in terms of accuracy and ROC-AUC score?**

- **Answer**: The Logistic Regression model provided the best ROC-AUC score for both red and white wines, indicating its superior performance compared to RandomForestClassifier and KNeighborsClassifier. The RandomForestClassifier and KNeighborsClassifier also performed well, but they did not match the Logistic Regression model's performance in terms of both accuracy and ROC-AUC.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for any enhancements or bug fixes.

## Acknowledgments

- **Akanksha Kushwaha** for project submission.
- **Scikit-learn Documentation** for guidance on model implementation and evaluation.
