import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys

def main(input_path, output_path):
    # Load the original dataset
    df = pd.read_csv('Glassdoor Gender Pay Gap.csv')

    # Define features and target from the original dataset
    X = df[['JobTitle', 'Gender', 'Age', 'PerfEval', 'Education', 'Dept', 'Seniority']]
    y = df['BasePay']

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

    # Train the model on the entire dataset
    model.fit(X, y)

    # Load the new input dataset (without BasePay and Bonus)
    input_df = pd.read_csv(input_path)

    # Ensure the input data has the same features as the original dataset
    assert set(input_df.columns) == set(X.columns), "Input dataset columns do not match the training features."

    # Predict BasePay for the new dataset
    predicted_basepay = model.predict(input_df)

    # Add the predicted BasePay to the new dataset
    input_df['BasePay'] = predicted_basepay

    # Save the new dataset with predicted BasePay to a CSV file
    input_df.to_csv(output_path, index=False)

    print(f"Predicted BasePay added to the dataset and saved to {output_path}")

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)

