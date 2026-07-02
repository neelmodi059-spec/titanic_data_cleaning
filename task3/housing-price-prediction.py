import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# load dataset
df = pd.read_csv('housing.csv')

print("dataset info like data types and non-null counts: "+'\n')
print(df.info())

print('-'*50)

print("total row and column in the dataset: "+'\n')
print(df.shape)

print('-'*50)

print("sum of null values in each column: "+'\n')
print(df.isnull().sum())

print('-'*50)

cols = ['mainroad', 'guestroom', 'basement',
        'hotwaterheating', 'airconditioning', 'prefarea']

for col in cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

le = LabelEncoder()
df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])

print(df.head())

print('-'*50)

X = df.drop(columns=['price'])   # saare features
y = df['price']                  # target

print("Features shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model training complete!")

y_pred = model.predict(X_test)

print("Predicted prices:", y_pred[:5])
print("Actual prices:", y_test.values[:5])

# ----

# mae batata hai ki average difference between predicted and actual values kitna hai.
mae = mean_absolute_error(y_test, y_pred)

# mse simple me kahe to average of squared differences between predicted and actual values hai. Ye outliers ko zyada penalize karta hai.
mse = mean_squared_error(y_test, y_pred)

# r2 
r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"R2   : {r2:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.tight_layout()
plt.show()

print('-'*50)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print(coef_df)

plt.figure(figsize=(10, 5))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Feature Coefficients')
plt.tight_layout()
plt.show()

