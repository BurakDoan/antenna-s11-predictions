import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_clean_data(file_path: str):
    data = pd.read_excel(file_path)
    df = data.copy()
    
    # Eksik değerleri ilk satırdaki sabitlerle doldurma
    for col in df.columns:
        df[col] = df[col].fillna(df[col].iloc[0])
        
    # İlk satırı ve gereksiz sütunları temizleme
    df = df.iloc[1:, 5:].drop_duplicates()
    
    # Sütun isimlerini standartlaştırma
    df.rename(columns={
        'l': 'patch_length',
        'w': 'patch_width',
        'lg': 'ground_length',
        'wg': 'ground_width',
        'g': 'gap',
        'S1,1-y ': 'S11_dB'
    }, inplace=True)
    
    X = df.drop(columns=['S11_dB'])
    y = df['S11_dB']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_features(X_train, X_test, save_scaler_path="models/scaler.pkl"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, save_scaler_path)
    return X_train_scaled, X_test_scaled, scaler