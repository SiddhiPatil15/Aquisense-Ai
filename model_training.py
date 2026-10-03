import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_and_compare_models(data_path):
    # 1. Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Assuming the target variable is 'Groundwater_Level' or similar
    # Using 'pH' as target for example if 'Groundwater_Level' is not present
    target_col = 'pH' if 'pH' in df.columns else df.columns[-1]
    
    # Preprocessing (Handle missing values, separate features/target)
    df = df.dropna()
    X = df.drop(columns=[target_col, 'Date', 'Well_ID'], errors='ignore')
    # Filter only numeric columns for features
    X = X.select_dtypes(include=[np.number])
    y = df[target_col]
    
    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Define models
    # Keeping existing models intact and adding new ones
    models = {
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
    }
    
    results = []
    
    # 4. Train and evaluate loop
    print("Training models...")
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        preds = model.predict(X_test_scaled)
        
        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        # Store results and model
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            '_model_obj': model
        })
        
    # 5. Create Comparison Table
    results_df = pd.DataFrame(results)
    
    # Sort by Best R2 (Highest)
    results_df = results_df.sort_values(by='R2', ascending=False).reset_index(drop=True)
    
    # Display table without the model object column
    print("\nModel Comparison Table:")
    print(results_df[['Model', 'MAE', 'RMSE', 'R2']].to_string(index=False))
    
    # 6. Select and save best model
    best_row = results_df.iloc[0]
    best_model_name = best_row['Model']
    best_model_obj = best_row['_model_obj']
    
    # Save best model
    model_filename = 'best_model.joblib'
    joblib.dump(best_model_obj, model_filename)
    
    # Print summary
    print(f"\nSummary:")
    print(f"{best_model_name} selected: highest R² of {best_row['R2']:.4f} and RMSE of {best_row['RMSE']:.4f}.")
    print(f"Saved the best model to '{model_filename}'.")

if __name__ == "__main__":
    # Point this to your actual dataset path
    dataset_path = 'data/sample/synthetic_groundwater_sample.csv'
    train_and_compare_models(dataset_path)
