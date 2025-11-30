import pandas as pd
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ChurnModelTrainer
from evaluation import ModelEvaluator

def main():
    """Complete Customer Churn Prediction Pipeline"""
    
    print("="*70)
    print("CUSTOMER CHURN PREDICTION PROJECT")
    print("="*70)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    trainer = ChurnModelTrainer()
    evaluator = ModelEvaluator()
    
    # 1. Load Data
    print("\n[STEP 1] Loading Data...")
    df = preprocessor.load_data('data/churn_data.csv')  # Update with your file path
    print(df.head())
    print(f"\nDataset Info:")
    print(df.info())
    
    # 2. Data Cleaning
    print("\n[STEP 2] Data Cleaning...")
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.remove_duplicates(df)
    
    # 3. Feature Engineering
    print("\n[STEP 3] Feature Engineering...")
    df = feature_engineer.create_features(df)
    
    # 4. Encode Categorical Variables
    print("\n[STEP 4] Encoding Categorical Variables...")
    df = preprocessor.encode_categorical(df, target_column='Churn')
    
    # 5. Prepare Train/Test Split
    print("\n[STEP 5] Preparing Train/Test Split...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, target_column='Churn')
    
    # 6. Scale Features
    print("\n[STEP 6] Scaling Features...")
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # 7. Handle Class Imbalance with SMOTE
    print("\n[STEP 7] Handling Class Imbalance...")
    X_train_balanced, y_train_balanced = feature_engineer.handle_imbalance_smote(
        X_train_scaled, y_train
    )
    
    # 8. Train Models
    print("\n[STEP 8] Training Models...")
    models = trainer.train_all_models(X_train_balanced, y_train_balanced)
    
    # 9. Evaluate Models
    print("\n[STEP 9] Evaluating Models...")
    for model_name, model in models.items():
        result = evaluator.evaluate_model(model, X_test_scaled, y_test, model_name)
        
        # Plot visualizations
        evaluator.plot_confusion_matrix(y_test, result['y_pred'], model_name)
        evaluator.plot_roc_curve(y_test, result['y_pred_proba'], model_name)
        evaluator.plot_precision_recall_curve(y_test, result['y_pred_proba'], model_name)
    
    # 10. Compare Models
    print("\n[STEP 10] Comparing Models...")
    comparison = evaluator.compare_models()
    
    # 11. Save Best Model
    best_model_name = comparison['roc_auc'].idxmax()
    best_model = models[best_model_name]
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    trainer.save_model(best_model, f'models/best_model_{best_model_name.replace(" ", "_")}.pkl')
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"ROC-AUC Score: {comparison.loc[best_model_name, 'roc_auc']:.4f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()