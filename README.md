# AI-AND-MACHINE-LEARNING
# multiple_regression_analysis.ipynb

# -----------------------------
# Multi-Channel Marketing Analysis
# -----------------------------

# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# -----------------------------
# Step 1: Load the Dataset
# -----------------------------
# Replace with your CSV file path if different
df = pd.read_csv('/mnt/data/c107fa55-45be-4f9c-8f61-088e88d1fb0c.csv')

# Quick overview
print("First 5 rows:")
display(df.head())
print("\nDataset Info:")
df.info()
print("\nSummary Statistics:")
display(df.describe())

# -----------------------------
# Step 2: Exploratory Data Analysis
# -----------------------------
# Pairplot to visualize relationships
sns.pairplot(df)
plt.show()

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# -----------------------------
# Step 3: Check for Multicollinearity
# -----------------------------
# Define independent variables
X = df[['TV', 'Radio', 'Social Media']]
# Add constant for VIF calculation
X_const = sm.add_constant(X)

vif = pd.DataFrame()
vif['Variable'] = X_const.columns
vif['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print("\nVariance Inflation Factor (VIF):")
display(vif)

# -----------------------------
# Step 4: Build Multiple Linear Regression Model
# -----------------------------
y = df['Sales']
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

print("\nModel Summary:")
print(model.summary())

# -----------------------------
# Step 5: Validate Regression Assumptions
# -----------------------------
# Residuals
residuals = model.resid
fitted = model.fittedvalues

# Linearity & Homoscedasticity
plt.scatter(fitted, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

# Normality
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()
sm.qqplot(residuals, line='45')
plt.title("QQ Plot of Residuals")
plt.show()

# -----------------------------
# Step 6: Interpret Coefficients
# -----------------------------
coef_df = pd.DataFrame({
    'Variable': model.params.index,
    'Coefficient': model.params.values,
    'P-Value': model.pvalues.values
})
print("\nCoefficients Interpretation:")
display(coef_df)

# Example interpretation
print("\nInterpretation Example:")
print("Holding Radio and Social Media spend constant, each additional $1K spent on TV is associated with a ${:.2f} increase in Sales.".format(model.params['TV']))

# -----------------------------
# Step 7: Business Recommendation
# -----------------------------
# Basic recommendation based on coefficient size and significance
significant_vars = coef_df[coef_df['P-Value'] < 0.05].sort_values(by='Coefficient', ascending=False)
print("\nPriority Marketing Spend Recommendations:")
for index, row in significant_vars.iterrows():
    print(f"- Focus on {row['Variable']}: estimated impact ${row['Coefficient']:.2f} per unit spend (p-value {row['P-Value']:.4f})")

# -----------------------------
# END OF ANALYSIS
# -----------------------------
