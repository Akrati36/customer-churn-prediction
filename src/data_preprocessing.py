import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load dataset from CSV"""
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        print(f"\nMissing values before:\n{df.isnull().sum()}")
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        print(f"\nMissing values after:\n{df.isnull().sum()}")
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        print(f"\nRemoved {before - after} duplicate rows")
        return df
    
    def encode_categorical(self, df, target_column='Churn'):
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != target_column]
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Encode target variable
        if target_column in df.columns:
            le_target = LabelEncoder()
            df[target_column] = le_target.fit_transform(df[target_column])
            self.label_encoders[target_column] = le_target
        
        print(f"\nEncoded {len(categorical_cols)} categorical columns")
        return df
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self, df, target_column='Churn', test_size=0.2, random_state=42):
        """Complete data preparation pipeline"""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Class distribution in train:\n{y_train.value_counts()}")
        
        return X_train, X_test, y_train, y_test