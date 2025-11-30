# 🚀 Quick Start Guide

Get your Customer Churn Prediction project running in 5 minutes!

## Step 1: Clone the Repository

```bash
git clone https://github.com/Akrati36/customer-churn-prediction.git
cd customer-churn-prediction
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Get the Dataset

### Option A: Use Kaggle Dataset (Recommended)

1. Download the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
2. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the `data/` folder
3. Rename it to `churn_data.csv`

```bash
# After downloading
mv WA_Fn-UseC_-Telco-Customer-Churn.csv data/churn_data.csv
```

### Option B: Use Your Own Dataset

Place your CSV file in `data/churn_data.csv` with:
- Customer features as columns
- A target column named `Churn` (values: Yes/No or 1/0)

## Step 4: Run the Pipeline

```bash
python main.py
```

This will:
- ✅ Clean and preprocess data
- ✅ Engineer features
- ✅ Train 3 ML models
- ✅ Generate evaluation plots
- ✅ Save the best model

**Expected runtime**: 2-5 minutes

## Step 5: View Results

Check the generated files:
- `confusion_matrix_*.png` - Classification matrices
- `roc_curve_*.png` - ROC curves
- `pr_curve_*.png` - Precision-Recall curves
- `model_comparison.png` - Model performance comparison
- `models/best_model_*.pkl` - Saved best model

## Step 6: Run the Web App (Optional)

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start making predictions!

## 🎯 Expected Output

```
======================================================================
CUSTOMER CHURN PREDICTION PROJECT
======================================================================

[STEP 1] Loading Data...
Dataset loaded: (7043, 21)

[STEP 2] Data Cleaning...
Missing values before:
...

[STEP 8] Training Models...
=== Training Logistic Regression ===
=== Training Random Forest ===
=== Training XGBoost ===

[STEP 10] Comparing Models...
======================================================================
MODEL COMPARISON
======================================================================
                      accuracy  precision  recall  f1_score  roc_auc
Logistic Regression     0.8012     0.6543  0.5234    0.5821   0.8456
Random Forest           0.8234     0.6891  0.5987    0.6401   0.8723
XGBoost                 0.8456     0.7123  0.6234    0.6645   0.8912

======================================================================
BEST MODEL: XGBoost
ROC-AUC Score: 0.8912
======================================================================
```

## 🐛 Troubleshooting

### Issue: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Issue: "File not found: data/churn_data.csv"
Make sure you've placed your dataset in the `data/` folder with the correct name.

### Issue: "Memory Error"
Reduce dataset size or use a machine with more RAM.

## 📚 Next Steps

1. **Tune Hyperparameters**: Modify `src/model_training.py`
2. **Add Features**: Edit `src/feature_engineering.py`
3. **Deploy**: Use Streamlit Cloud, Heroku, or Railway
4. **Experiment**: Try different algorithms and techniques

## 💡 Tips

- Start with the default Kaggle dataset to ensure everything works
- Check the plots to understand model performance
- Experiment with different SMOTE strategies
- Use the Streamlit app for interactive predictions

## 🆘 Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review the code comments in each module
- Open an issue on GitHub

---

Happy predicting! 🎉