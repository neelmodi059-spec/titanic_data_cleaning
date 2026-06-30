import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('titanic.csv')

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

print("-"*50)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)

print(df.isnull().sum())

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Outliers before removal
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(df['Age'])
plt.title('Age - Before')


plt.subplot(1, 2, 2)
sns.boxplot(df['Fare'])
plt.title('Fare - Before')


plt.suptitle('Before Outlier Removal')
plt.tight_layout()
plt.show()


Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5*IQR) & (df['Fare'] <= Q3 + 1.5*IQR)]

Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Age'] >= Q1 - 1.5*IQR) & (df['Age'] <= Q3 + 1.5*IQR)]


# Outliers after removal
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(df['Age'])
plt.title('Age - After')

plt.subplot(1, 2, 2)
sns.boxplot(df['Fare'])
plt.title('Fare - After')

plt.suptitle('After Outlier Removal')
plt.tight_layout()
plt.show()

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print(df.head())

df.to_csv('titanic cleaned.csv', index=False)
print("done")
