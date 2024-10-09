# Titanic Survival Prediction Project

## Project Overview
This project aims to predict the survival chances of Titanic passengers based on various features such as age, sex, fare, and class using the Titanic dataset. The project includes data cleaning, visualization, machine learning models, and a graphical user interface (GUI) for making predictions.

## Features
- **Data Cleaning & Preparation**: The dataset has missing values, duplicates, and outliers, which were handled before applying machine learning models.
- **Data Visualization**: Various plots were created to show survival distributions by features like age, sex, fare, etc. (Graphs will be added below).
- **Machine Learning Models**:
  - Logistic regression model using raw data.
  - Logistic regression model using percentage-based survival rates for certain features like age group, sex, fare, etc.
- **GUI for Survival Prediction**: A user-friendly GUI was created using Tkinter to input passenger data (age, sex, fare, etc.) and predict the survival chance.

## Data Processing
- Filled missing values in the `age` column using the median.
- Removed columns such as `ticket`, `cabin`, and `embarked` to focus on essential features.
- Grouped continuous features (age, fare) into bins for better interpretability.
- Added new features like family size and whether the passenger was traveling alone.

## Visualizations
- Survival rate by sex, age, class, fare, and family size.
- Boxplots to identify and remove fare outliers.

![SurvivalRateByFeatures](https://github.com/user-attachments/assets/e13ae7be-6100-4a61-9205-bdb9d0946f4a)
![Outlier](https://github.com/user-attachments/assets/0aa6e640-784f-45f0-b439-e91f1b5eb93a)
![SurvivalAgeRate](https://github.com/user-attachments/assets/46944685-0e22-4f0e-9a06-8fbcff7113ca)
![SurvivalPClassRate](https://github.com/user-attachments/assets/ee683249-8967-499b-a849-e25e7404a2a7)
![SurvivalFareRate](https://github.com/user-attachments/assets/1d15ded9-1f68-4b01-b4c2-3b8c0e954190)
![SurvivalFamilySizeRate](https://github.com/user-attachments/assets/83ce3a40-1eea-4303-937d-24dc43296398)


## Machine Learning Models

- **Model 1**: Logistic Regression with polynomial features (raw data).
- **Model 2**: Logistic Regression using percentage-based survival rates for features.

After cross-validation and hyperparameter tuning, both models were compared. Despite the expectation that the model based on percentage data would perform better, the results were comparable.

### Model Accuracy

- **Raw Data Model:**
  - DegreeRD: 1, AccuracyRD: 0.7706
  - DegreeRD: 2, AccuracyRD: 0.8211
  - DegreeRD: 3, AccuracyRD: 0.8028
  - DegreeRD: 4, AccuracyRD: 0.7798
  - DegreeRD: 5, AccuracyRD: 0.7661
  - DegreeRD: 6, AccuracyRD: 0.7569
  - DegreeRD: 7, AccuracyRD: 0.7661
  - DegreeRD: 8, AccuracyRD: 0.7661
  - DegreeRD: 9, AccuracyRD: 0.7569
  - DegreeRD: 10, AccuracyRD: 0.7569
  
  **Best Degree**: 2, **Best Accuracy**: 0.8211  
  **Final Raw Data Model Accuracy**: 0.8211

- **Percentage Data Model:**
  - DegreeSR: 1, AccuracySR: 0.7890
  - DegreeSR: 2, AccuracySR: 0.7661
  - DegreeSR: 3, AccuracySR: 0.7798
  - DegreeSR: 4, AccuracySR: 0.7798
  - DegreeSR: 5, AccuracySR: 0.7798
  - DegreeSR: 6, AccuracySR: 0.8028
  - DegreeSR: 7, AccuracySR: 0.7982
  - DegreeSR: 8, AccuracySR: 0.7936
  - DegreeSR: 9, AccuracySR: 0.7936
  - DegreeSR: 10, AccuracySR: 0.7936
  
  **Best Degree**: 6, **Best Accuracy**: 0.8028  
  **Final Survival Rate Model Accuracy**: 0.8028

## GUI Application

The GUI allows users to input passenger information (age, sex, class, etc.) and predict the survival probability. It was built using the Tkinter library and features validation for inputs to ensure correct data entry.

![Screenshot 2024-10-09 144731](https://github.com/user-attachments/assets/5d4fbd64-6152-4343-8527-7e65a0667214)


## Requirements

- Python 3.x
- Libraries: `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `tkinter`, `PIL (Pillow)`
