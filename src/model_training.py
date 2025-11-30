from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

class ChurnModelTrainer:
    def __init__(self):
        self.models = {}
    
    def train_logistic_regression(self, X_train, y_train, class_weight='balanced'):
        """Train Logistic Regression model"""
        print("\n=== Training Logistic Regression ===")
        
        lr_model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        )
        
        lr_model.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr_model
        
        print("Logistic Regression trained successfully")
        return lr_model
    
    def train_random_forest(self, X_train, y_train, class_weight='balanced'):
        """Train Random Forest model"""
        print("\n=== Training Random Forest ===")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        self.models['Random Forest'] = rf_model
        
        print("Random Forest trained successfully")
        return rf_model
    
    def train_xgboost(self, X_train, y_train, scale_pos_weight=None):
        """Train XGBoost model"""
        print("\n=== Training XGBoost ===")
        
        # Calculate scale_pos_weight if not provided
        if scale_pos_weight is None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count
        
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        print("XGBoost trained successfully")
        return xgb_model
    
    def train_all_models(self, X_train, y_train):
        """Train all models"""
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        
        return self.models
    
    def save_model(self, model, filename):
        """Save trained model"""
        joblib.dump(model, filename)
        print(f"\nModel saved to {filename}")
    
    def load_model(self, filename):
        """Load saved model"""
        model = joblib.load(filename)
        print(f"\nModel loaded from {filename}")
        return model