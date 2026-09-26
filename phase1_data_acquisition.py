import pandas as pd
import numpy as np

def load_and_inspect_data(filepath):
    """
    Phase 1: Data Acquisition and Initial Inspection
    This function loads a CSV dataset and prints out basic information 
    necessary to understand its structure before cleaning.
    """
    print(f"--- Loading data from {filepath} ---")
    
    try:
        # 1. Load the dataset
        df = pd.read_csv(filepath)
        print("Data successfully loaded!\n")
        
        # 2. Inspect the shape (rows, columns)
        print(f"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns.\n")
        
        # 3. View the first 5 rows
        print("--- First 5 Rows ---")
        
        # Checking if standard Telco dataset columns exist to show a cleaner preview
        display_cols = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'] 
        if set(display_cols).issubset(df.columns):
            print(df[display_cols].head())
        else:
            print(df.head())
            
        print("\n--- Data Types and Missing Values ---")
        # 4. Check data types and look for obvious missing values
        print(df.info())
        
        print("\n--- Basic Summary Statistics ---")
        # 5. Get basic stats for numerical columns
        print(df.describe())
        
        return df

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please ensure you have downloaded the dataset (e.g., Telco Customer Churn from Kaggle) and placed it in the correct directory.")
        return None

if __name__ == "__main__":
    # TODO: Download the dataset and place it in the same folder as this script.
    # We highly recommend the 'Telco Customer Churn' dataset from Kaggle for this project:
    # https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    
    DATA_PATH = "Bank_Churn_Classification_Dataset.csv" 
    
    # Run the acquisition phase
    raw_data = load_and_inspect_data(DATA_PATH)
