import pandas as pd
import statsmodels.api as sm

df = pd.read_excel('cars.xls')

print(df.head())

# Create a code for the model name
df['Model_ord'] = pd.Categorical(df.Model).codes

print(df.head())

x = df[['Mileage', 'Model_ord', 'Doors']]
y = df[['Price']]


X1 = sm.add_constant(x)
est = sm.OLS(y, X1).fit()

print(est.summary())
