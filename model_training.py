import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

def train_and_compare_models(data_path):
    # 1. Load data
    print(f"Loading hydrological dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Priority target is Groundwater_Level, fallback to pH or last column
    if 'Groundwater_Level' in df.columns:
        target_col = 'Groundwater_Level'
    elif 'pH' in df.columns:
        target_col = 'pH'
    else:
        target_col = df.columns[-1]
        
    print(f"Target Variable: [{target_col}]")
    
    # Feature Engineering: Rolling climate indicators
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by=['Well_ID', 'Date']).reset_index(drop=True)
    
    if 'Rainfall' in df.columns:
        df['Rainfall_4wk_Rolling'] = df.groupby('Well_ID')['Rainfall'].transform(lambda s: s.rolling(4, min_periods=1).mean())
    if 'Temperature' in df.columns:
        df['Temp_4wk_Rolling'] = df.groupby('Well_ID')['Temperature'].transform(lambda s: s.rolling(4, min_periods=1).mean())

    # Preprocessing (Handle missing values, separate features/target)
    df = df.dropna()
    X = df.drop(columns=[target_col, 'Date', 'Well_ID'], errors='ignore')
    # Filter only numeric columns for features
    X = X.select_dtypes(include=[np.number])
    y = df[target_col]
    feature_names = X.columns.tolist()
    
    # 2. Time-ordered Train/Test split per well (80% train, 20% test)
    if 'Well_ID' in df.columns:
        train_indices, test_indices = [], []
        for _, group in df.groupby('Well_ID'):
            k = int(len(group) * 0.8)
            if k > 0 and len(group) - k > 0:
                train_indices.extend(group.index[:k])
                test_indices.extend(group.index[k:])
            else:
                train_indices.extend(group.index)
        if train_indices and test_indices:
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        else:
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    else:
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Define models (including XGBoost and Tree Ensembles)
    models = {
        'XGBoost Regressor': xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = []
    
    # 4. Train and evaluate loop
    print("Training models and computing metrics...")
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
    results_df = results_df.sort_values(by='RMSE', ascending=True).reset_index(drop=True)
    
    print("\n" + "="*50)
    print("  AquiSense AI Model Comparison Leaderboard")
    print("="*50)
    print(results_df[['Model', 'MAE', 'RMSE', 'R2']].to_string(index=False))
    
    # 6. Select and save best model
    best_row = results_df.iloc[0]
    best_model_name = best_row['Model']
    best_model_obj = best_row['_model_obj']
    
    # Feature Importance for Explainable AI
    if hasattr(best_model_obj, 'feature_importances_'):
        fi = best_model_obj.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': fi}).sort_values('Importance', ascending=False)
        print("\nExplainable AI (XAI) — Top Feature Attributions:")
        print(fi_df.head(6).to_string(index=False))
    
    # Save best model
    model_filename = 'best_model.joblib'
    joblib.dump({
        'model': best_model_obj,
        'model_name': best_model_name,
        'scaler': scaler,
        'features': feature_names,
        'target': target_col,
        'metrics': {'MAE': best_row['MAE'], 'RMSE': best_row['RMSE'], 'R2': best_row['R2']}
    }, model_filename)
    
    print(f"\n[SUCCESS] {best_model_name} selected as optimal model (RMSE: {best_row['RMSE']:.4f}, R²: {best_row['R2']:.4f}).")
    print(f"Saved artifacts to '{model_filename}'.")

if __name__ == "__main__":
    import os
    dataset_path = 'data/sample/cgwb_groundwater_records.csv'
    if not os.path.exists(dataset_path):
        dataset_path = 'data/sample/synthetic_groundwater_sample.csv'
    train_and_compare_models(dataset_path)
