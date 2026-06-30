# Titanic Dataset - EDA Task

This project is part of my AI and ML Internship Task 2. Here I performed 
Exploratory Data Analysis on the Titanic dataset to find patterns using 
stats and graphs.

## Steps I Followed

1. Checked basic statistics of the dataset like mean, median, std using describe().
2. Plotted histograms for numeric columns to understand their distribution.
3. Used boxplots to compare Age and Fare against the Survived column.
4. Built a correlation matrix and heatmap to check relation between features.
5. Used pairplot to compare multiple columns together visually.
6. Analyzed survival count based on Sex and Pclass.

## Key Observations

- Females survived more compared to males.
- Passengers in 1st class had better survival chances than 3rd class.
- Fare distribution was skewed, only a few passengers paid very high amounts.
- No strong correlation was found between Age and Survived.

## Files in Repo

- eda_titanic.py — main analysis script
- train.csv — original dataset
- README.md — project details

## Libraries Used

Pandas, Matplotlib, Seaborn
