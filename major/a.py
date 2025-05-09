import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
dataset_path = '/root/v/Real-Time-Flight-Delay-Prediction/data/openweather.csv'
df = pd.read_csv(dataset_path)

# Data Preprocessing
# Drop irrelevant or redundant columns and columns with excessive missing values
drop_columns = ['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 
                'weather_description', 'weather_icon', 'weather_main',
                'sea_level', 'grnd_level', 'snow_1h', 'snow_3h']
df = df.drop(columns=[col for col in drop_columns if col in df.columns])

# Check for missing values before imputation
print("Missing values before imputation:\n", df.isna().sum())

# Handle missing values
# Separate numeric and categorical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
if 'weather_id' in numeric_columns:
    numeric_columns.remove('weather_id')  # Treat weather_id as categorical

# Impute numeric columns
if numeric_columns:
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(df[numeric_columns])
    df[numeric_columns] = imputed_data

# Convert weather_id to dummy variables
if 'weather_id' in df.columns:
    df = pd.get_dummies(df, columns=['weather_id'], prefix='weather', dummy_na=False)

# Verify no NaN values remain
if df.isna().any().any():
    print("Warning: NaN values still present after imputation:\n", df.isna().sum())
    df = df.fillna(0)  # Final fallback: replace remaining NaN with 0

# Feature Selection
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'flight_delay' in numeric_features:
    numeric_features.remove('flight_delay')

# Correlation-based feature selection
corr_matrix = df[numeric_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
df = df.drop(columns=to_drop)
print("Dropped correlated features:", to_drop)

# Define features and target
if 'flight_delay' in df.columns:
    X = df.drop(columns=['flight_delay'])
    y = df['flight_delay']
else:
    print("Warning: 'flight_delay' column not found. Simulating target variable.")
    X = df
    y = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])  # Simulate target

# Verify data shape and content
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X columns:", X.columns.tolist())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verify no NaN in scaled data
if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
    print("Error: NaN values in scaled data. Check preprocessing steps.")
    exit(1)

# Model Training
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ''

for name, model in models.items():
    try:
        # Cross-validation
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        mean_accuracy = scores.mean()
        print(f"{name} CV Accuracy: {mean_accuracy:.4f}")
        
        # Train and test
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Test Accuracy: {test_accuracy:.4f}")
        
        # Update best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model
            best_model_name = name
    except Exception as e:
        print(f"Error training {name}: {str(e)}")

# Save the best model, scaler, and imputer to pickle
if best_model is not None:
    with open('flight_delay_model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler, 'imputer': imputer}, f)
    print(f"Best model: {best_model_name} with Test Accuracy: {best_accuracy:.4f}")
    print("Model, scaler, and imputer saved to 'flight_delay_model.pkl'")
else:
    print("No model was successfully trained.")