# Exploratory Data Analysis Script
# Convert this to a Jupyter notebook for interactive exploration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_explore_data(filepath='../data/churn_data.csv'):
    """Load and perform initial exploration"""
    print("="*70)
    print("EXPLORATORY DATA ANALYSIS - CUSTOMER CHURN")
    print("="*70)
    
    # Load data
    df = pd.read_csv(filepath)
    
    print(f"\n1. DATASET OVERVIEW")
    print(f"   Shape: {df.shape}")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    
    # Display first few rows
    print(f"\n2. FIRST 5 ROWS:")
    print(df.head())
    
    # Data types
    print(f"\n3. DATA TYPES:")
    print(df.dtypes)
    
    # Missing values
    print(f"\n4. MISSING VALUES:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   No missing values!")
    
    # Basic statistics
    print(f"\n5. NUMERICAL FEATURES STATISTICS:")
    print(df.describe())
    
    return df

def analyze_target_variable(df, target='Churn'):
    """Analyze the target variable distribution"""
    print(f"\n6. TARGET VARIABLE ANALYSIS ({target}):")
    
    # Value counts
    counts = df[target].value_counts()
    print(f"\n   Distribution:")
    print(counts)
    
    # Percentages
    percentages = df[target].value_counts(normalize=True) * 100
    print(f"\n   Percentages:")
    print(percentages)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    counts.plot(kind='bar', ax=axes[0], color=['green', 'red'])
    axes[0].set_title(f'{target} Distribution (Count)')
    axes[0].set_xlabel(target)
    axes[0].set_ylabel('Count')
    
    # Pie chart
    percentages.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                     colors=['green', 'red'], startangle=90)
    axes[1].set_ylabel('')
    axes[1].set_title(f'{target} Distribution (Percentage)')
    
    plt.tight_layout()
    plt.savefig('target_distribution.png')
    print(f"\n   Plot saved: target_distribution.png")

def analyze_numerical_features(df, target='Churn'):
    """Analyze numerical features"""
    print(f"\n7. NUMERICAL FEATURES ANALYSIS:")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numerical_cols:
        numerical_cols.remove(target)
    
    print(f"   Found {len(numerical_cols)} numerical features")
    
    # Distribution plots
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols):
        df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    # Hide empty subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('numerical_distributions.png')
    print(f"   Plot saved: numerical_distributions.png")

def analyze_categorical_features(df, target='Churn'):
    """Analyze categorical features"""
    print(f"\n8. CATEGORICAL FEATURES ANALYSIS:")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target in categorical_cols:
        categorical_cols.remove(target)
    
    print(f"   Found {len(categorical_cols)} categorical features")
    
    for col in categorical_cols[:5]:  # Show first 5
        print(f"\n   {col}:")
        print(df[col].value_counts())

def correlation_analysis(df, target='Churn'):
    """Analyze correlations"""
    print(f"\n9. CORRELATION ANALYSIS:")
    
    # Select numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Correlation matrix
    corr_matrix = numerical_df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print(f"   Plot saved: correlation_heatmap.png")
    
    # Top correlations with target (if numerical)
    if target in numerical_df.columns:
        target_corr = corr_matrix[target].sort_values(ascending=False)
        print(f"\n   Top correlations with {target}:")
        print(target_corr)

def churn_analysis_by_features(df, target='Churn'):
    """Analyze churn rates by different features"""
    print(f"\n10. CHURN RATE BY FEATURES:")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target in categorical_cols:
        categorical_cols.remove(target)
    
    # Analyze top 4 categorical features
    for col in categorical_cols[:4]:
        churn_rate = df.groupby(col)[target].value_counts(normalize=True).unstack()
        print(f"\n   Churn rate by {col}:")
        print(churn_rate)

def main():
    """Run complete EDA"""
    # Load data
    df = load_and_explore_data()
    
    # Analyze target
    analyze_target_variable(df)
    
    # Analyze features
    analyze_numerical_features(df)
    analyze_categorical_features(df)
    
    # Correlations
    correlation_analysis(df)
    
    # Churn analysis
    churn_analysis_by_features(df)
    
    print("\n" + "="*70)
    print("EXPLORATORY ANALYSIS COMPLETE!")
    print("Check the generated PNG files for visualizations")
    print("="*70)

if __name__ == "__main__":
    main()