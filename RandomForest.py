import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
df = pd.read_csv('Glassdoor Gender Pay Gap.csv')

# Define features and target from the dataset
X = df[['JobTitle', 'Gender', 'Age', 'PerfEval', 'Education', 'Dept', 'Seniority']]
y = df['BasePay']  # Or df['Bonus']

# Preprocessing pipeline for numerical features
numeric_features = ['Age', 'PerfEval', 'Seniority']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical features
categorical_features = ['JobTitle', 'Gender', 'Education', 'Dept']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that combines preprocessing and model training
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
mean_rmse = np.mean(rmse_scores)

print(f'Cross-Validation RMSE: {mean_rmse:.2f}')

# Train the model on the entire training data
model.fit(X, y)

# Predict on the test set
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mean_y = np.mean(y)
percentage_rmse = (rmse / mean_y) * 100

print(f'Percentage RMSE for Random Forest: {percentage_rmse:.2f}%')

# Feature importances from the Random Forest model
importances = model.named_steps['regressor'].feature_importances_

# Get the feature names after one-hot encoding
feature_names = (numeric_features + 
                 list(model.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .named_steps['onehot']
                      .get_feature_names_out(categorical_features)))

# Create a DataFrame to display feature importances
importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Display the impact of gender on salary
gender_importance = importances_df[importances_df['Feature'].str.contains('Gender')]
print(gender_importance)

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

print(f'Predicted salary for male: {predicted_salary_male[0]:.2f}')
print(f'Predicted salary for female: {predicted_salary_female[0]:.2f}')