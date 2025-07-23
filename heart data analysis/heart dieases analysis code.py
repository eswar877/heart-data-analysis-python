import sys
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

def sanitize_filename(s):
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)

##############################
# PREPROCESSING FOR HEART CSV
##############################
def preprocess_heart_data(df):
    df.columns = [c.strip() for c in df.columns]

    # Drop irrelevant columns if any
    drop_cols = ["Unnamed: 0", "ID"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Identify target
    target_col = "target"
    if target_col not in df.columns:
        raise ValueError("Expected target column 'target' not found.")

    # Handle non-numeric columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    return df

##########################
# DESCRIPTIVE ANALYSIS
##########################
def descriptive_analysis(df):
    print("\n====== DESCRIPTIVE ANALYSIS ======\n")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print(df.describe(include='all').T)
    print("\nValue counts for each column:\n")
    for col in df.columns:
        print(f"{col}:\n{df[col].value_counts()}\n")

##########################
# EDA WITH CHARTS
##########################
def eda_charts(df, output_dir="eda_charts_heart"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nEDA charts will be saved in: {output_dir}/\n")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        if col == "target":
            continue

        # Histogram
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{sanitize_filename(col)}_hist.png")
        plt.close()

        # Boxplot
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{sanitize_filename(col)}_box.png")
        plt.close()

    # Correlation heatmap
    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()

##########################
# PREDICTIVE ANALYSIS
##########################
def predictive_analysis(df):
    print("\n====== PREDICTIVE ANALYSIS ======\n")
    target_col = "target"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"Random Forest Classifier - Accuracy: {acc:.2f}\n")
    print(report)
    print("Confusion Matrix:\n", cm)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("Top features influencing heart disease prediction:\n")
    print(feature_importances.sort_values(ascending=False).head(7))
    return feature_importances

##########################
# PRESCRIPTIVE ANALYSIS
##########################
def prescriptive_analysis(feature_importances):
    print("\n====== PRESCRIPTIVE ANALYSIS ======\n")
    high_impact = feature_importances.sort_values(ascending=False).head(3).index.tolist()
    print(f"Top 3 influential features: {', '.join(high_impact)}")
    print("Recommendation: Focus on monitoring and improving these factors for better heart health prediction.")

##########################
# MAIN
##########################
def main():
    # Replace this with sys.argv[1] if using CLI
    fname = "heart.csv"

    print(f"Loading data from {fname} ...\n")
    df = pd.read_csv(fname)

    df_processed = preprocess_heart_data(df)
    print("Preprocessing complete. Columns after preprocessing:\n")
    print(list(df_processed.columns))

    descriptive_analysis(df_processed)
    eda_charts(df_processed)
    feature_importances = predictive_analysis(df_processed)

    if feature_importances is not None:
        prescriptive_analysis(feature_importances)

if __name__ == "__main__":
    main()