import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def run_ml_pipeline(filepath):
    print("--- Phase 2: Data Cleaning & Preprocessing ---")
    df = pd.read_csv(filepath)
    
    # Drop useless ID columns that shouldn't be fed into a machine learning model
    columns_to_drop = ['Unnamed: 0', 'CustomerID']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print(f"Dropped columns: {columns_to_drop}")
    
    print("\n--- Phase 3: Exploratory Data Analysis & Visualizations ---")
    plt.figure(figsize=(15, 5))
    
    # 1. Pie Chart for Churn Distribution
    plt.subplot(1, 3, 1)
    df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
    plt.title('Churn Distribution (Target Variable)')
    plt.ylabel('') # Remove y-axis label for cleaner pie chart
    
    # 2. Histplot for Tenure Distribution
    plt.subplot(1, 3, 2)
    sns.histplot(df['Tenure'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Customer Tenure (Months)')
    
    # 3. Boxplot for Monthly Charges vs Churn
    # This helps us see if higher monthly charges lead to higher churn
    plt.subplot(1, 3, 3)
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
    plt.title('Monthly Charges vs Churn')
    
    plt.tight_layout()
    plt.savefig('churn_visualizations.png')
    print("Visualizations saved as 'churn_visualizations.png'")
    
    print("\n--- Phase 4: Feature Engineering (One-Hot Encoding) ---")
    # Identify categorical columns (strings)
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"Applying One-Hot Encoding to categorical variables: {list(categorical_cols)}")
    
    # One-hot encoding converts text categories (like 'Male'/'Female') into 1s and 0s
    # drop_first=True prevents the dummy variable trap
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("\n--- Setting up X and y (Train / Test Split) ---")
    # y is what we want to predict. X is all the features used to predict it.
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Splitting data: 80% for training the model, 20% for testing it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training dataset (X_train) shape: {X_train.shape}")
    print(f"Testing dataset (X_test) shape: {X_test.shape}")
    
    print("\n--- Scaling numerical features ---")
    # Models like Logistic Regression work best when all numbers are on the same scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- Phase 5: Machine Learning Classification ---")
    # Using Random Forest, which is a powerful classification algorithm
    print("Training a Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred))
    
    return model, df_encoded

if __name__ == "__main__":
    DATA_PATH = "Bank_Churn_Classification_Dataset.csv"
    
    try:
        model, processed_data = run_ml_pipeline(DATA_PATH)
    except FileNotFoundError:
        print("Dataset not found! Make sure 'Bank_Churn_Classification_Dataset.csv' is in the same folder.")
    except ImportError as e:
        print(f"Missing a required library: {e}")
        print("Please run this command in your terminal: pip install pandas matplotlib seaborn scikit-learn")
