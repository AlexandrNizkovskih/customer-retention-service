import os, joblib, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, \
    f1_score, precision_score, recall_score, accuracy_score

# Читаем конфиг, чтобы понимать целевую колонку и путь к модели
import yaml
CFG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
MODEL_PATH = CFG.get("model_path", "model/model.pkl")
TARGET_COL = CFG.get("target_col", "Class")  # для churn: Churn
DROP_COLS  = CFG.get("drop_cols", [])
THRESHOLD  = float(CFG.get("threshold", 0.5))

# Пытаемся авто-угадать путь к данным
# Для churn/fraud обычно есть явный таргет в train.csv/исходном CSV,
# для sentiment таргет формируется в train.py  тогда этот скрипт пропустит авто-оценку.
cand_paths = ["data/train.csv", "data/creditcard.csv", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"]
DATA_PATH = None
for p in cand_paths:
    if os.path.exists(p):
        DATA_PATH = p
        break

if DATA_PATH is None:
    print("Не найден data/train.csv (или известный файл). Укажите DATA_PATH вручную или адаптируйте скрипт.")
    raise SystemExit(0)

df = pd.read_csv(DATA_PATH)
if TARGET_COL not in df.columns:
    print(f"TARGET_COL='{TARGET_COL}' не найден в {DATA_PATH}. Для наборов без явного таргета (напр., Sentiment140) пропускаем.")
    raise SystemExit(0)

y = df[TARGET_COL]

# конвертируем бинарный target к 0/1
if y.dtype == object or str(y.dtype).startswith("category"):
    uniq = sorted(y.unique())
    if set(uniq) == {"No","Yes"}:
        y = (y == "Yes").astype(int)
    else:
        # любая другая строковая бинарка → первая категория = 0, вторая = 1
        mapping = {val:i for i,val in enumerate(uniq)}
        y = y.map(mapping).astype(int)

X = df.drop(columns=[c for c in [TARGET_COL] + DROP_COLS if c in df.columns])

pipe = joblib.load(MODEL_PATH)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
proba = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_te)

# Метрики
fpr, tpr, _ = roc_curve(y_te, proba)
prec, rec, _ = precision_recall_curve(y_te, proba)
roc_auc = auc(fpr, tpr)
pr_auc  = average_precision_score(y_te, proba)
y_hat   = (proba >= THRESHOLD).astype(int)

f1  = f1_score(y_te, y_hat)
p   = precision_score(y_te, y_hat)
r   = recall_score(y_te, y_hat)
acc = accuracy_score(y_te, y_hat)

os.makedirs("report", exist_ok=True)

# ROC
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
plt.savefig("report/roc_curve.png", dpi=150); plt.close()

# PR
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={pr_auc:.3f})")
plt.savefig("report/pr_curve.png", dpi=150); plt.close()

# Гистограмма вероятностей
plt.figure()
plt.hist(proba, bins=50)
plt.xlabel("proba"); plt.ylabel("count"); plt.title("Predicted probabilities")
plt.savefig("report/prob_hist.png", dpi=150); plt.close()

with open("report/metrics.md","w",encoding="utf-8") as f:
    f.write("# Отчёт по качеству\n\n")
    f.write("## Итоги\n")
    f.write(f"- ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"- PR-AUC:  {pr_auc:.4f}\n")
    f.write(f"- F1:      {f1:.4f} (порог = {THRESHOLD})\n")
    f.write(f"- Precision / Recall: {p:.4f} / {r:.4f}\n")
    f.write(f"- Accuracy: {acc:.4f}\n\n")
    f.write("## Графики\n")
    f.write("- roc_curve.png  ROC-кривая\n")
    f.write("- pr_curve.png   PR-кривая\n")
    f.write("- prob_hist.png  гистограмма вероятностей\n")

print("Готово: report/roc_curve.png, report/pr_curve.png, report/prob_hist.png и report/metrics.md")