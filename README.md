# Customer Churn Prediction 🎯

A comprehensive machine learning project to predict customer churn using Python, Scikit-learn, and XGBoost.

## 📋 Project Overview

This project implements a complete ML pipeline for predicting customer churn with:
- Advanced data preprocessing and cleaning
- Feature engineering techniques
- Class imbalance handling using SMOTE
- Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Comprehensive evaluation metrics

## 🚀 Features

- **Data Preprocessing**: Missing value handling, duplicate removal, encoding
- **Feature Engineering**: Custom feature creation, scaling, normalization
- **Imbalance Handling**: SMOTE oversampling for balanced training
- **Multiple Models**: Compare 3 different ML algorithms
- **Evaluation**: ROC-AUC, Confusion Matrix, Precision-Recall curves
- **Visualization**: Automated plot generation for all metrics

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Akrati36/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt
```

## 🗂️ Project Structure

```
customer-churn-prediction/
├── data/                      # Dataset directory
│   └── churn_data.csv        # Your dataset (add your own)
├── src/                       # Source code modules
│   ├── data_preprocessing.py # Data cleaning and preparation
│   ├── feature_engineering.py # Feature creation and SMOTE
│   ├── model_training.py     # Model training logic
│   └── evaluation.py         # Evaluation metrics and plots
├── models/                    # Saved trained models
├── main.py                    # Main pipeline script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 💻 Usage

### 1. Prepare Your Dataset

Place your customer churn dataset in the `data/` directory as `churn_data.csv`. The dataset should have:
- Customer features (demographics, usage patterns, etc.)
- A target column named `Churn` (Yes/No or 1/0)

**Recommended Dataset**: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) from Kaggle

### 2. Run the Pipeline

```bash
python main.py
```

This will:
1. Load and clean the data
2. Engineer features
3. Handle class imbalance
4. Train all models
5. Generate evaluation metrics and plots
6. Save the best model

### 3. View Results

The script generates:
- Confusion matrices for each model
- ROC curves with AUC scores
- Precision-Recall curves
- Model comparison chart
- Saved best model in `models/` directory

## 🤖 Models Implemented

### 1. Logistic Regression
- Baseline linear model
- Fast training and inference
- Good interpretability

### 2. Random Forest
- Ensemble tree-based model
- Handles non-linear relationships
- Feature importance analysis

### 3. XGBoost
- Gradient boosting algorithm
- High performance
- Handles imbalanced data well

## 📊 Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to churn cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification breakdown
- **Precision-Recall Curve**: Trade-off visualization

## 🔧 Customization

### Modify Feature Engineering

Edit `src/feature_engineering.py` to add custom features:

```python
def create_features(self, df):
    # Add your custom features here
    df['new_feature'] = df['col1'] / df['col2']
    return df
```

### Adjust Model Parameters

Edit `src/model_training.py` to tune hyperparameters:

```python
rf_model = RandomForestClassifier(
    n_estimators=200,  # Increase trees
    max_depth=15,      # Deeper trees
    # ... other parameters
)
```

### Change Imbalance Strategy

Edit `main.py` to use different sampling:

```python
# Use combined sampling instead of SMOTE
X_train_balanced, y_train_balanced = feature_engineer.handle_imbalance_combined(
    X_train_scaled, y_train
)
```

## 📈 Expected Results

With proper dataset and tuning, you should achieve:
- **ROC-AUC**: 0.80 - 0.90
- **Accuracy**: 75% - 85%
- **Recall**: 70% - 85% (important for churn detection)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: ML algorithms and preprocessing
- **XGBoost**: Gradient boosting
- **Imbalanced-learn**: SMOTE implementation
- **Matplotlib/Seaborn**: Visualization

## 📝 Next Steps

- [ ] Add hyperparameter tuning (GridSearchCV)
- [ ] Implement cross-validation
- [ ] Create Streamlit web app for predictions
- [ ] Add feature importance analysis
- [ ] Deploy model as REST API
- [ ] Add more ensemble methods

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Akrati Mishra**
- GitHub: [@Akrati36](https://github.com/Akrati36)

## 🙏 Acknowledgments

- Dataset: Kaggle Telco Customer Churn
- Inspired by real-world churn prediction challenges
- Built for learning and portfolio demonstration

---

⭐ Star this repo if you find it helpful!