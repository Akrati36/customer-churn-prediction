import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class FeatureEngineer:
    def __init__(self):
        pass
    
    def create_features(self, df):
        """Create new features"""
        # Example feature engineering (customize based on your dataset)
        
        # Tenure groups
        if 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, np.inf],
                                        labels=['0-1 year', '1-2 years', '2-4 years', 
                                               '4-5 years', '5+ years'])
        
        # Total charges per month
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df['charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Service usage score
        service_cols = [col for col in df.columns if 'Service' in col or 'service' in col]
        if service_cols:
            df['total_services'] = df[service_cols].sum(axis=1)
        
        print(f"\nCreated new features. Total columns: {df.shape[1]}")
        return df
    
    def handle_imbalance_smote(self, X_train, y_train, sampling_strategy='auto'):
        """Handle class imbalance using SMOTE"""
        print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
        return X_resampled, y_resampled
    
    def handle_imbalance_combined(self, X_train, y_train):
        """Handle imbalance using combined over and under sampling"""
        over = SMOTE(sampling_strategy=0.5, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
        
        pipeline = Pipeline([('over', over), ('under', under)])
        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
        
        print(f"\nAfter combined sampling: {pd.Series(y_resampled).value_counts().to_dict()}")
        return X_resampled, y_resampled
    
    def get_class_weights(self, y_train):
        """Calculate class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        print(f"\nClass weights: {class_weights}")
        return class_weights