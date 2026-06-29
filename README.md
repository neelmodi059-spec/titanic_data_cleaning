# Titanic Dataset - Data Cleaning and Preprocessing

This project is part of my AI and ML Internship Task 1. The goal was to take raw data 
and clean it properly before using it in any machine learning model.

---

## About the Dataset

- Dataset Name: Titanic
- File: train.csv
- Downloaded from: Kaggle
- Rows: 891
- Columns: 12

Titanic dataset has information about passengers like their age, sex, ticket class, 
fare, cabin, where they boarded from and most importantly whether they survived or not.

---

## Why Data Cleaning is Needed

Real world data is messy. It contains empty values, useless columns, text data and 
extreme values called outliers. If we directly give this dirty data to a ML model 
the results will be wrong. So cleaning the data first is very important.

---

## Process I Followed

### 1) Explored the Data
After loading the dataset I checked how the data looks using head(), info() and 
describe() functions. Then I checked which columns have missing values using isnull().sum().

Missing values found:
- Age — 177 missing
- Cabin — 687 missing
- Embarked — 2 missing

### 2) Fixed Missing Values
- Age column: filled empty values with the average age of all passengers
- Embarked column: filled empty values with the most common value which was S
- Cabin column: dropped the whole column because 77% data was missing in it

### 3) Dropped Useless Columns
Columns like Name, Ticket and PassengerId do not give any useful information 
to the model so I removed them from the dataset.

### 4) Converted Text to Numbers
ML models cannot read text so I converted text columns to numbers.

- Sex column: used Label Encoding — female = 0, male = 1
- Embarked column: used One Hot Encoding — created separate columns for S, C, Q

### 5) Found Outliers using Boxplot
I plotted boxplots for Age and Fare columns. From the graph I could clearly see 
some extreme values that were far away from the rest of the data. These are outliers.

### 6) Removed Outliers using IQR
I calculated IQR which is Q3 minus Q1 and removed all values that were outside 
the range of Q1 - 1.5 times IQR and Q3 + 1.5 times IQR.

After removing I plotted the boxplot again to confirm outliers were gone.

### 7) Scaled the Features
Age values were between 0 to 80 and Fare values were between 0 to 500. 
This big difference in range can affect model performance.

So I applied StandardScaler on both columns to bring them to a common scale 
where mean is 0 and standard deviation is 1.

### 8) Saved Clean Data
At the end I saved the final cleaned data as titanic_cleaned.csv file.

---

## Files in this Repo

- train.csv — raw original dataset
- Data_preprocessing.py — python code for all cleaning steps
- titanic_cleaned.csv — cleaned dataset ready for ML
- README.md — project explanation

---

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Key Things I Learned

- How to find and fix missing values
- Difference between Label Encoding and One Hot Encoding
- How to detect outliers visually using boxplot
- How to remove outliers using IQR method
- How StandardScaler works and why scaling is needed
- Overall importance of preprocessing in machine learning
