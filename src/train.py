import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_pipeline import load_and_clean_data, preprocess_features

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def train_pipeline():
    mlflow.set_experiment("Antenna_S11_Optimization")
    
    X_train, X_test, y_train, y_test = load_and_clean_data("data/dataset_anten.xlsx")
    X_train_scaled, X_test_scaled, _ = preprocess_features(X_train, X_test)
    
    params = {
        "n_estimators": 2000,
        "max_samples": 0.9,
        "min_samples_split": 2,
        "random_state": 7
    }
    
    with mlflow.start_run(run_name="RandomForest_Tuned"):
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train) # Tree bazlı modelde ölçeklenmemiş veri de kullanılabilir
        
        predictions = model.predict(X_test)
        rmse, mae, r2 = evaluate_metrics(y_test, predictions)
        
        # Log Params & Metrics
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log Model
        mlflow.sklearn.log_model(model, "rf_antenna_model")
        
        # Yerel Kayıt
        joblib.dump(model, "models/best_model.pkl")
        print(f"Model Eğitildi ve Kaydedildi. RMSE: {rmse:.4f}")

if __name__ == "__main__":
    train_pipeline()