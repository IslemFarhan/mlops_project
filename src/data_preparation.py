import pandas as pd
import numpy as np
import os


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")


def drop_null(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()



def preprocess_health_data(df):
    # Mapping for Diabetes
    diabetes_mapping = {
        'No': 0, 
        'No, pre-diabetes or borderline diabetes': 0, 
        'Yes, but female told only during pregnancy': 1,
        'Yes': 1
    }
    df['Diabetes'] = df['Diabetes'].map(diabetes_mapping)

    # Encoding Sex: 0 for female, 1 for male
    df['Sex'] = df['Sex'].astype(str).str.strip()

    df['Sex'] = df['Sex'].map({'Female': 0, 'Male': 1})
    # Convert remaining categorical variables with "Yes" and "No" values to binary format for correlation computation
    binary_columns = ['Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History', 'Exercise']
    
    for column in binary_columns:
        df[column] = df[column].map({'Yes': 1, 'No': 0})

    # Ordinal encoding for General_Health, Age_Category, BMI_Category
    general_health_mapping = {
        'Poor': 0,
        'Fair': 1,
        'Good': 2,
        'Very Good': 3,
        'Excellent': 4
    }
    df['General_Health'] = df['General_Health'].map(general_health_mapping)

   

    age_category_mapping = {
        '18-24': 0,
        '25-29': 1,
        '30-34': 2,
        '35-39': 3,
        '40-44': 4,
        '45-49': 5,
        '50-54': 6,
        '55-59': 7,
        '60-64': 8,
        '65-69': 9,
        '70-74': 10,
        '75-79': 11,
        '80+': 12
    }
    df['Age_Category'] = df['Age_Category'].map(age_category_mapping)
     # Ordinal encoding for Checkup
    checkup_mapping = {
        'Never': 0,
        '5 or more years ago': 1,
        'Within the past 5 years': 2,
        'Within the past 2 years': 3,
        'Within the past year': 4
    }
    df['Checkup'] = df['Checkup'].map(checkup_mapping)


    return df






def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")


def main():
    try:
        raw_data_path = "./data/raw/"
        processed_data_path = "./data/processed"

        
      

        selected_column=['General_Health', 'Checkup', 'Exercise', 'Heart_Disease', 'Skin_Cancer',
          'Other_Cancer', 'Diabetes', 'Arthritis', 'Sex',
          'Age_Category', 'Height_(cm)', 'Weight_(kg)', 'BMI', 'Smoking_History',
          'Alcohol_Consumption', 'Fruit_Consumption',
          'Green_Vegetables_Consumption']
        

        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))

        # Drop NaN values
        train_data = drop_null(train_data)
        test_data = drop_null(test_data)

        train_data=preprocess_health_data(train_data)
        test_data=preprocess_health_data(test_data)

        train_data=train_data[selected_column]
        test_data=test_data[selected_column]


        # Create processed data directory if it doesn't exist
        os.makedirs(processed_data_path, exist_ok=True)

        # Save the processed data
        save_data(train_data, os.path.join(processed_data_path, "train_processed.csv"))
        save_data(test_data, os.path.join(processed_data_path, "test_processed.csv"))

    except Exception as e:
        raise Exception(f"An error occurred: {e}")


if __name__ == "__main__":

    main()