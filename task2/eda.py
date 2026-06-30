import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("D:/extra_folder/AIML INTERNS/Task-1/titanic_cleaned.csv")
# print(df.head())

print('\n',"_"*50,'\n')

# print(df.describe())

print('\n',"_"*50,'\n')

# print(df['Survived'].value_counts())

print('\n',"_"*50,'\n')

# df.hist(figsize=(12, 10), bins=20)
# plt.suptitle('Histograms of Numeric Features')
# plt.tight_layout()
# plt.show()

print('\n',"_"*50,'\n')

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# sns.boxplot(x='Survived', y='Age', data=df)
# plt.title('Age vs Survived')

# plt.subplot(1, 2, 2)
# sns.boxplot(x='Survived', y='Fare', data=df)
# plt.title('Fare vs Survived')

# plt.tight_layout()
# plt.show()

print('\n',"_"*50,'\n')

# numeric_df = df.select_dtypes(include=[np.number])

# plt.figure(figsize=(10, 8))
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix')
# plt.show()

print('\n',"_"*50,'\n')

# sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']] , hue='Survived')
# plt.show()

print('\n',"_"*50,'\n')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')

plt.subplot(1, 2, 2)
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Class')

plt.tight_layout()
plt.show()