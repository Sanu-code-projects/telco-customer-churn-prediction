#!/usr/bin/env python
# coding: utf-8

# # Predicting Customer Churn and Identifying Retention Drivers Using Data Mining Techniques
# 
# Source: Kaggle — https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
# 

# ## 1. Environment and reproducibility
# 
# 
# 

# In[3]:


import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

RANDOM_STATE = 42


# ## 2. Data loading
# 
# 

# In[4]:


candidate_paths = [
    "WA_Fn-UseC_-Telco-Customer-Churn(1).csv"
]

dataset_path = next((p for p in candidate_paths if os.path.exists(p)), None)

if dataset_path is None:
    matches = glob.glob("/content/*Telco*Churn*.csv") + glob.glob("*Telco*Churn*.csv")
    dataset_path = matches[0] if matches else None

if dataset_path is None:
    raise FileNotFoundError(
        "Telco Customer Churn CSV not found. Upload the Kaggle CSV to the notebook environment and run this cell again."
    )

df = pd.read_csv(dataset_path)

print("Dataset path:", dataset_path)
print("Dataset shape:", df.shape)
display(df.head())


# ## 3. Initial data understanding and quality assessment
# 
# 

# In[5]:


print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
display(df.dtypes.to_frame("dtype"))

print("\nMissing values reported by pandas:")
display(df.isna().sum().to_frame("missing_values"))

print("\nDuplicated rows:", df.duplicated().sum())
print("Duplicated customer IDs:", df["customerID"].duplicated().sum())

print("\nUnique values per column:")
display(df.nunique().sort_values().to_frame("unique_values"))


# In[6]:


blank_total_charges = df["TotalCharges"].astype(str).str.strip().eq("").sum()

print("Blank strings in TotalCharges:", blank_total_charges)
print("\nChurn counts:")
display(df["Churn"].value_counts().to_frame("count"))

print("\nChurn percentages:")
display((df["Churn"].value_counts(normalize=True) * 100).round(2).to_frame("percentage"))


# ## 4. Data cleaning
# 
# 

# In[7]:


data = df.copy()

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

missing_total = data.loc[data["TotalCharges"].isna(), ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]

print("Missing TotalCharges after numeric conversion:", data["TotalCharges"].isna().sum())
display(missing_total)


# In[8]:


if len(missing_total) > 0:
    print("Tenure values among records with missing TotalCharges:")
    display(missing_total["tenure"].value_counts().sort_index().to_frame("count"))

data = data.drop_duplicates().reset_index(drop=True)

print("Shape after duplicate removal:", data.shape)
print("Remaining missing values:")
display(data.isna().sum().sort_values(ascending=False).head(10).to_frame("missing_values"))


# ## 5. Exploratory data analysis
# 
# 

# In[9]:


fig, ax = plt.subplots(figsize=(6, 4))
churn_counts = data["Churn"].value_counts().reindex(["No", "Yes"])
ax.bar(churn_counts.index, churn_counts.values)
ax.set_title("Figure 1. Customer Churn Distribution")
ax.set_xlabel("Churn")
ax.set_ylabel("Number of Customers")
for i, value in enumerate(churn_counts.values):
    ax.text(i, value, f"{value}\n({value / len(data):.1%})", ha="center", va="bottom")
plt.tight_layout()
plt.show()


# In[10]:


def churn_rate_table(column):
    table = (
        data.groupby(column, dropna=False)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
        .rename("ChurnRatePercent")
        .to_frame()
    )
    table["CustomerCount"] = data[column].value_counts(dropna=False)
    return table

for column in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]:
    print(f"\nChurn rate by {column}")
    display(churn_rate_table(column).round(2))


# In[11]:


plot_columns = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]

for idx, column in enumerate(plot_columns, start=2):
    rates = (
        data.groupby(column)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rates.index.astype(str), rates.values)
    ax.set_title(f"Figure {idx}. Churn Rate by {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.show()


# In[12]:


fig, ax = plt.subplots(figsize=(8, 4))
for churn_value in ["No", "Yes"]:
    subset = data.loc[data["Churn"] == churn_value, "tenure"]
    ax.hist(subset, bins=24, alpha=0.55, label=churn_value)
ax.set_title("Figure 6. Tenure Distribution by Churn Status")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Frequency")
ax.legend(title="Churn")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
groups = [
    data.loc[data["Churn"] == "No", "MonthlyCharges"],
    data.loc[data["Churn"] == "Yes", "MonthlyCharges"]
]
ax.boxplot(groups, labels=["No", "Yes"])
ax.set_title("Figure 7. Monthly Charges by Churn Status")
ax.set_xlabel("Churn")
ax.set_ylabel("Monthly Charges")
plt.tight_layout()
plt.show()


# In[13]:


numerical_summary = (
    data.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]]
    .agg(["mean", "median", "std"])
    .round(2)
)

display(numerical_summary)


# ## 6. Statistical association analysis
# 
# 

# In[14]:


def cramers_v(contingency_table):
    chi2 = chi2_contingency(contingency_table, correction=False)[0]
    n = contingency_table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denominator = min(k_corr - 1, r_corr - 1)
    return np.sqrt(phi2_corr / denominator) if denominator > 0 else 0

categorical_for_tests = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

association_results = []

for column in categorical_for_tests:
    table = pd.crosstab(data[column], data["Churn"])
    chi2, p_value, dof, expected = chi2_contingency(table)
    association_results.append({
        "Feature": column,
        "ChiSquare": chi2,
        "DegreesOfFreedom": dof,
        "PValue": p_value,
        "CramersV": cramers_v(table)
    })

association_df = pd.DataFrame(association_results).sort_values("CramersV", ascending=False)
display(association_df.round(5))


# ## 7. Feature engineering
# 
# 

# In[15]:


service_columns = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

def engineer_features(frame):
    result = frame.copy()
    if "TotalCharges" in result.columns:
        result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")

    result["tenure_group"] = pd.cut(
        result["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
    )

    available_services = [c for c in service_columns if c in result.columns]
    result["TotalServices"] = result[available_services].eq("Yes").sum(axis=1)

    if "customerID" in result.columns:
        result = result.drop(columns="customerID")

    return result

model_data = engineer_features(data)

print("Model-ready shape before target split:", model_data.shape)
display(model_data.head())


# In[16]:


fig, ax = plt.subplots(figsize=(8, 4))
tenure_rates = (
    model_data.groupby("tenure_group", observed=False)["Churn"]
    .apply(lambda s: (s == "Yes").mean() * 100)
)
ax.bar(tenure_rates.index.astype(str), tenure_rates.values)
ax.set_title("Figure 8. Churn Rate Across Customer Tenure Groups")
ax.set_xlabel("Tenure Group")
ax.set_ylabel("Churn Rate (%)")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
service_rates = (
    model_data.groupby("TotalServices")["Churn"]
    .apply(lambda s: (s == "Yes").mean() * 100)
)
ax.plot(service_rates.index, service_rates.values, marker="o")
ax.set_title("Figure 9. Churn Rate by Number of Services")
ax.set_xlabel("Total Services")
ax.set_ylabel("Churn Rate (%)")
plt.tight_layout()
plt.show()


# ## 8. Machine-learning formulation and train/test split
# 
# 
# 

# In[17]:


X = model_data.drop(columns="Churn")
y = model_data["Churn"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("\nTraining churn rate:", round(y_train.mean(), 4))
print("Test churn rate:", round(y_test.mean(), 4))


# ## 9. Leakage-safe preprocessing pipeline
# 
# 
# 

# In[18]:


numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Numerical features:", numeric_features)
print("\nCategorical features:", categorical_features)


# ## 10. Baseline and candidate models
# 
# 
# 

# In[19]:


models = {
    "Dummy Baseline": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_rows = []

for name, estimator in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_rows.append({
        "Model": name,
        "CV Accuracy": scores["test_accuracy"].mean(),
        "CV Precision": scores["test_precision"].mean(),
        "CV Recall": scores["test_recall"].mean(),
        "CV F1": scores["test_f1"].mean(),
        "CV ROC-AUC": scores["test_roc_auc"].mean()
    })

cv_results = pd.DataFrame(cv_rows).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)
display(cv_results.round(4))


# In[20]:


fig, ax = plt.subplots(figsize=(9, 4))
comparison = cv_results[cv_results["Model"] != "Dummy Baseline"].set_index("Model")["CV ROC-AUC"].sort_values()
ax.barh(comparison.index, comparison.values)
ax.set_title("Figure 10. Cross-Validated ROC-AUC by Model")
ax.set_xlabel("Mean CV ROC-AUC")
ax.set_xlim(max(0.5, comparison.min() - 0.05), min(1.0, comparison.max() + 0.05))
plt.tight_layout()
plt.show()


# ## 11. Hyperparameter tuning
# 
# 

# In[21]:


substantive_results = cv_results[cv_results["Model"] != "Dummy Baseline"].copy()
best_initial_name = substantive_results.iloc[0]["Model"]

model_lookup = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
}

param_grids = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"]
    },
    "Decision Tree": {
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_split": [2, 10, 25],
        "model__min_samples_leaf": [1, 5, 10],
        "model__class_weight": [None, "balanced"]
    },
    "Random Forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 8, 14],
        "model__min_samples_leaf": [1, 3, 8],
        "model__max_features": ["sqrt", 0.6],
        "model__class_weight": [None, "balanced"]
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [1, 2, 3],
        "model__subsample": [0.8, 1.0]
    }
}

tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model_lookup[best_initial_name])
])

grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grids[best_initial_name],
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Best initial model:", best_initial_name)
print("Best CV ROC-AUC after tuning:", round(grid_search.best_score_, 4))
print("Best hyperparameters:")
display(pd.Series(grid_search.best_params_, name="value").to_frame())


# ## 12. Final evaluation on the unseen test set
# 
# 

# In[22]:


y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

test_metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    "Score": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0),
        roc_auc_score(y_test, y_prob)
    ]
})

display(test_metrics.round(4))

print("Classification report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], digits=4))


# In[23]:


fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Churn", "Churn"],
    cmap=None,
    ax=ax
)
ax.set_title("Figure 11. Confusion Matrix of the Tuned Best Model")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_prob, name=best_initial_name, ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_title("Figure 12. ROC Curve of the Tuned Best Model")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=best_initial_name, ax=ax)
ax.set_title("Figure 13. Precision-Recall Curve of the Tuned Best Model")
plt.tight_layout()
plt.show()


# ## 13. Test-set comparison of all substantive models
# 
# 

# In[24]:


test_comparison_rows = []

for name, estimator in model_lookup.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clone(estimator))
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    test_comparison_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })

test_comparison = pd.DataFrame(test_comparison_rows).sort_values("ROC-AUC", ascending=False)
display(test_comparison.round(4))


# ## 14. Model interpretation with permutation importance
# 

# In[25]:


perm = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": perm.importances_mean,
    "Std": perm.importances_std
}).sort_values("Importance", ascending=False)

display(importance_df.head(15).round(5))

top_importance = importance_df.head(12).sort_values("Importance")

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_importance["Feature"], top_importance["Importance"])
ax.set_title("Figure 14. Permutation Feature Importance")
ax.set_xlabel("Decrease in ROC-AUC after Feature Permutation")
plt.tight_layout()
plt.show()


# ## 15. Business-oriented knowledge discovery
# 
# 
# 

# In[26]:


business_features = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "PaperlessBilling",
    "SeniorCitizen",
    "tenure_group"
]

business_tables = {}

for feature in business_features:
    summary = (
        model_data.groupby(feature, observed=False)
        .agg(
            Customers=("Churn", "size"),
            Churned=("Churn", lambda s: (s == "Yes").sum()),
            ChurnRate=("Churn", lambda s: (s == "Yes").mean())
        )
        .reset_index()
    )
    summary["ChurnRate"] = summary["ChurnRate"] * 100
    summary = summary.sort_values(["ChurnRate", "Customers"], ascending=[False, False])
    business_tables[feature] = summary

    print(f"\n{feature}")
    display(summary.round(2))


# In[27]:


top_segments = []

for feature, summary in business_tables.items():
    valid = summary[summary["Customers"] >= 50].copy()
    if not valid.empty:
        row = valid.iloc[0]
        top_segments.append({
            "Feature": feature,
            "HighestRiskSegment": row[feature],
            "Customers": int(row["Customers"]),
            "ChurnRatePercent": row["ChurnRate"]
        })

top_segments_df = pd.DataFrame(top_segments).sort_values("ChurnRatePercent", ascending=False)
display(top_segments_df.round(2))


# ## 16. Sample individual prediction
# 
# 

# In[28]:


sample_customer = pd.DataFrame([{
    "customerID": "SAMPLE-0001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 477.50
}])

sample_features = engineer_features(sample_customer)
sample_features = sample_features[X_train.columns]

sample_prediction = best_model.predict(sample_features)[0]
sample_probability = best_model.predict_proba(sample_features)[0, 1]

print("Sample customer prediction:", "Churn" if sample_prediction == 1 else "No Churn")
print("Predicted churn probability:", f"{sample_probability:.2%}")
display(sample_customer)


# ## 17. Predictions for a sample of unseen test customers
# 

# In[29]:


sample_test = X_test.head(10).copy()
sample_test_results = sample_test.copy()
sample_test_results["ActualChurn"] = y_test.loc[sample_test.index].map({0: "No", 1: "Yes"})
sample_test_results["PredictedChurn"] = best_model.predict(sample_test)
sample_test_results["PredictedChurn"] = sample_test_results["PredictedChurn"].map({0: "No", 1: "Yes"})
sample_test_results["ChurnProbability"] = best_model.predict_proba(sample_test)[:, 1]

display(
    sample_test_results[
        ["tenure", "Contract", "InternetService", "MonthlyCharges",
         "TotalServices", "ActualChurn", "PredictedChurn", "ChurnProbability"]
    ].round(4)
)


# ## 18. Reproducible summary for the report
# 
# 

# In[30]:


print("DATASET SUMMARY")
print("Rows:", len(data))
print("Original columns:", df.shape[1])
print("Overall churn rate:", f"{(data['Churn'].eq('Yes').mean() * 100):.2f}%")
print("Blank/missing TotalCharges identified:", blank_total_charges)

print("\nMODEL SELECTION")
print("Best initial model:", best_initial_name)
print("Best tuned CV ROC-AUC:", f"{grid_search.best_score_:.4f}")

print("\nFINAL UNSEEN TEST METRICS")
for _, row in test_metrics.iterrows():
    print(f"{row['Metric']}: {row['Score']:.4f}")

print("\nTOP PERMUTATION IMPORTANCE FEATURES")
print(importance_df.head(10)[["Feature", "Importance"]].round(5).to_string(index=False))

