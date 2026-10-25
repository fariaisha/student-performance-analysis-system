import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Student_Performance.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).drop_duplicates()

target = "final_grade"
X = df.drop(columns=[target])
y = df[target]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_cols)
])

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=5, random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", C=2.0, probability=True, random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(
        f"{name}: accuracy={accuracy_score(y_test, pred):.4f}, "
        f"precision={precision_score(y_test, pred, average='weighted', zero_division=0):.4f}, "
        f"recall={recall_score(y_test, pred, average='weighted', zero_division=0):.4f}, "
        f"f1={f1_score(y_test, pred, average='weighted', zero_division=0):.4f}"
    )
    filename = {
        "Decision Tree": "decision_tree.pkl",
        "Gaussian Naive Bayes": "naive_bayes.pkl",
        "SVM": "svm.pkl"
    }[name]
    joblib.dump(pipe, os.path.join(MODEL_DIR, filename))

print("All three classification models trained successfully.")
