# House Price Prediction using Linear Regression

This project is from my AI and ML Internship Task 3. The goal was to predict 
house prices using a Linear Regression model trained on the Housing dataset.

## About the Dataset

- Dataset: Housing Price Dataset
- Source: Kaggle
- File used: Housing.csv

The dataset has features like area, number of bedrooms, bathrooms, stories,
parking, airconditioning, furnishing status and more.

## Steps I Followed

1. Imported the dataset and explored it using head() and info().
2. Handled yes/no text columns by converting them to 1 and 0.
3. Used Label Encoding for the furnishingstatus column.
4. Divided data into training set (80%) and testing set (20%).
5. Applied Linear Regression model from sklearn library.
6. Generated predictions on the test set.
7. Measured model performance using MAE, MSE and R2 score.
8. Created a scatter plot of Actual vs Predicted prices.
9. Visualized feature coefficients using a bar chart.

## What I Observed

- Higher area directly leads to higher price as expected.
- R2 score showed how well the model fits the data.
- Some features like hotwaterheating had less impact on price.

## Files in this Repo

- linear_regression.py — python script with all steps
- Housing.csv — raw dataset used for training
- README.md — project overview

## Tools and Libraries

Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn