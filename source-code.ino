pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap lime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Data cleaning and preprocessing
def preprocess_data(df):
    # Drop any duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Encode categorical variables (if any)
    df = pd.get_dummies(df, drop_first=True)

    return df

# Split the data into features and target
def split_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

# Normalize the data
def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/ckd_data.csv')

# Show first 5 rows of data
df.head()

# Check for missing values
df.isnull().sum()

# Descriptive statistics
df.describe()

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Distribution of the target variable
sns.countplot(x='ckd', data=df)
plt.title('CKD Distribution')
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Load and preprocess data
df = pd.read_csv('data/ckd_data.csv')
df = preprocess_data(df)

# Split data into features and target
X, y = split_data(df, target_column='ckd')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Support Vector Machine
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate models

# Random Forest Evaluation
print("Random Forest - Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# XGBoost Evaluation
print("XGBoost - Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# Support Vector Machine Evaluation
print("SVM - Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# ROC Curve - Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_rf))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='best')
plt.show()

# Feature Importance - Random Forest
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()
import shap

# Create the SHAP explainer object
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test)

# SHAP Dependence Plot
shap.dependence_plot('age', shap_values, X_test)
import lime
from lime.lime_tabular import LimeTabularExplainer

# Initialize the Lime explainer
explainer = LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=X.columns)

# Choose a random instance to explain
idx = 15
exp = explainer.explain_instance(X_test[idx], rf_model.predict_proba)

# Display the explanation
exp.show_in_notebook()
pandas==1.5.3
numpy==1.24.0
scikit-learn==1.2.1
xgboost==1.7.4
matplotlib==3.7.0
seaborn==0.12.2
shap==0.41.0
lime==0.3.3.1
