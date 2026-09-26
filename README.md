--- Loading data from Bank_Churn_Classification_Dataset.csv ---
Data successfully loaded!

Dataset Shape: 10000 rows and 10 columns.

--- First 5 Rows ---
   Unnamed: 0  CustomerID  Gender  SeniorCitizen  ...        Contract     PaymentMethod Churn TotalCharges
0           0           0    Male              0  ...        Two year      Mailed check     0      6153.40
1           1           1  Female              1  ...        Two year  Electronic check     0      2113.20
2           2           2    Male              0  ...        One year  Electronic check     0      4397.82
3           3           3    Male              1  ...  Month-to-month      Mailed check     1      1345.96
4           4           4    Male              1  ...        Two year      Mailed check     0       757.35

[5 rows x 10 columns]

--- Data Types and Missing Values ---
<class 'pandas.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 10 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Unnamed: 0      10000 non-null  int64  
 1   CustomerID      10000 non-null  int64  
 2   Gender          10000 non-null  str    
 3   SeniorCitizen   10000 non-null  int64  
 4   Tenure          10000 non-null  int64  
 5   MonthlyCharges  10000 non-null  float64
 6   Contract        10000 non-null  str    
 7   PaymentMethod   10000 non-null  str    
 8   Churn           10000 non-null  int64
 9   TotalCharges    10000 non-null  float64
dtypes: float64(2), int64(5), str(3)
memory usage: 781.4 KB
None

--- Basic Summary Statistics ---
        Unnamed: 0   CustomerID  SeniorCitizen        Tenure  MonthlyCharges         Churn  TotalCharges
count  10000.00000  10000.00000   10000.000000  10000.000000    10000.000000  10000.000000  10000.000000
mean    4999.50000   4999.50000       0.499300     35.955000       70.451038      0.267000   2541.807390
std     2886.89568   2886.89568       0.500025     20.501761       28.935692      0.442414   1879.645307
min        0.00000      0.00000       0.000000      1.000000       20.000000      0.000000     21.200000
25%     2499.75000   2499.75000       0.000000     18.000000       45.527500      0.000000   1035.057500
50%     4999.50000   4999.50000       0.000000     36.000000       70.585000      0.000000   2117.135000
75%     7499.25000   7499.25000       1.000000     54.000000       95.612500      1.000000   3717.352500
max     9999.00000   9999.00000       1.000000     71.000000      120.000000      1.000000   8384.390000


--- Phase 2: Data Cleaning & Preprocessing ---
Dropped columns: ['Unnamed: 0', 'CustomerID']

--- Phase 3: Exploratory Data Analysis & Visualizations ---
c:\Users\acer\Downloads\churn_project\phase2_to_4_modeling.py:35: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
Visualizations saved as 'churn_visualizations.png'

--- Phase 4: Feature Engineering (One-Hot Encoding) ---
c:\Users\acer\Downloads\churn_project\phase2_to_4_modeling.py:44: Pandas4Warning: For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated and will be removed in a future version. Explicitly pass 'str' to `include` to select them, or to `exclude` to remove them and silence this warning.
See https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#string-migration-select-dtypes for details on how to write code that works with pandas 2 and 3.
  categorical_cols = df.select_dtypes(include=['object']).columns
Applying One-Hot Encoding to categorical variables: ['Gender', 'Contract', 'PaymentMethod']

--- Setting up X and y (Train / Test Split) ---
Training dataset (X_train) shape: (8000, 10)
Testing dataset (X_test) shape: (2000, 10)

--- Scaling numerical features ---

--- Phase 5: Machine Learning Classification ---
Training a Random Forest Classifier...

--- Model Evaluation Results ---
Accuracy: 69.65%

Classification Report (Precision, Recall, F1-Score):
              precision    recall  f1-score   support

           0       0.74      0.91      0.82      1472
           1       0.28      0.10      0.14       528
           1       0.28      0.10      0.14       528

    accuracy                           0.70      2000
           1       0.28      0.10      0.14       528

    accuracy                           0.70      2000
   macro avg       0.51      0.50      0.48      2000
weighted avg       0.62      0.70      0.64      2000

           1       0.28      0.10      0.14       528

    accuracy                           0.70      2000
   macro avg       0.51      0.50      0.48      2000
weighted avg       0.62      0.70      0.64      2000
           1       0.28      0.10      0.14       528

    accuracy                           0.70      2000
   macro avg       0.51      0.50      0.48      2000
           1       0.28      0.10      0.14       528

           1       0.28      0.10      0.14       528

    accuracy                           0.70      2000
   macro avg       0.51      0.50      0.48      2000
weighted avg       0.62      0.70      0.64      2000
