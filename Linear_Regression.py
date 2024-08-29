import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
df = pd.read_csv('./Glassdoor Gender Pay Gap.csv')

# Define features and target from the dataset
X = df[['JobTitle', 'Gender', 'Age', 'PerfEval', 'Education', 'Dept', 'Seniority']]
y = df['BasePay']  # Or df['Bonus']

# Preprocessing pipeline for numerical features
# Pipeline Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# SimpleImputer Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
# StandardScaler Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
numeric_features = ['Age', 'PerfEval', 'Seniority']
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])


# Preprocessing pipeline for categorical features
# Pipeline Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# SimpleImputer Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
# OneHotEncoder Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
categorical_features = ['JobTitle', 'Gender', 'Education', 'Dept']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
# ColumnTransformer Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])

# Split the data (80% for training, 20% for validation)
# Train_test_split Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that combines preprocessing and model training
# # Pipeline Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', LinearRegression())])

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
# Mean Squared Error Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mean_y = np.mean(y_test)
percentage_rmse = (rmse / mean_y) * 100

print(f'Percentage RMSE for Linear Regression: {percentage_rmse}%')


# Extract the coefficients
coefficients = model.named_steps['regressor'].coef_

# Get the feature names after one-hot encoding
feature_names = (numeric_features + list(model.named_steps['preprocessor'].named_transformers_['cat'] .named_steps['onehot'].get_feature_names_out(categorical_features)))

# Create a DataFrame to display coefficients
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Display the impact of gender on salary
gender_coef = coef_df[coef_df['Feature'].str.contains('Gender')]
print(gender_coef)

# Example input data
example_data = pd.DataFrame({
    'JobTitle': ['Data Scientist'],
    'Gender': ['Male'],  # Change to 'Female' for comparison
    'Age': [30],
    'PerfEval': [4],
    'Education': ['Master'],
    'Dept': ['IT'],
    'Seniority': [5]
})

# Preprocess the example data
preprocessed_data = preprocessor.transform(example_data)

# Predict salary for male
predicted_salary_male = model.named_steps['regressor'].predict(preprocessed_data)

# Change gender to female
example_data['Gender'] = 'Female'

# Preprocess the example data again
preprocessed_data = preprocessor.transform(example_data)

# Predict salary for female
predicted_salary_female = model.named_steps['regressor'].predict(preprocessed_data)

print(f'Predicted salary for male: {predicted_salary_male[0]}')
print(f'Predicted salary for female: {predicted_salary_female[0]}')
