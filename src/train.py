
import os, sys, argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--target", default="Churn")
    parser.add_argument("--output", default="model/model.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # целевая: Yes/No -> 1/0
    df[args.target] = (df[args.target].astype(str).str.strip().str.upper() == "YES").astype(int)

    # простая очистка: выбросим customerID, пустые TotalCharges -> NaN -> заполним медианой позже scaler'ом
    drop_cols = [c for c in ["customerID"] if c in df.columns]
    y = df[args.target].astype(int)
    X = df.drop(columns=drop_cols + [args.target])

    # числовые/категориальные
    num_cols = X.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    pipe = Pipeline([
        ("prep", pre),
        ("clf", model),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe.fit(X_tr, y_tr)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    y_hat = (y_prob >= 0.5).astype(int)

    print(f"ROC-AUC: {roc_auc_score(y_te, y_prob):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_te, y_prob):.4f}")
    print(f"F1:      {f1_score(y_te, y_hat):.4f}")
    print(f"Prec:    {precision_score(y_te, y_hat):.4f}")
    print(f"Recall:  {recall_score(y_te, y_hat):.4f}")
    print(f"Acc:     {accuracy_score(y_te, y_hat):.4f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipe, args.output)
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
