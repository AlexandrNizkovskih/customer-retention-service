import argparse, os, yaml, joblib, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--config", default="config.yaml")
p.add_argument("--input", required=True)
p.add_argument("--output", default="predictions.csv")
args = p.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

MODEL_PATH = CFG.get("model_path", "model/model.pkl")
TARGET_COL = CFG.get("target_col")
DROP_COLS  = CFG.get("drop_cols", [])
TEXT_COL   = CFG.get("text_col")
THRESHOLD  = float(CFG.get("threshold", 0.5))
TOP_K      = int(CFG.get("top_k", 0))

pipe = joblib.load(MODEL_PATH)
df = pd.read_csv(args.input)

# подготовка входа
cols_to_drop = list(DROP_COLS)
if TARGET_COL and TARGET_COL in df.columns:
    cols_to_drop.append(TARGET_COL)
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

if TEXT_COL:
    if df.shape[1] == 1 and TEXT_COL not in df.columns:
        df.columns = [TEXT_COL]

# предикт
if hasattr(pipe, "predict_proba"):
    proba = pipe.predict_proba(df)[:, 1]
    out = pd.DataFrame({"proba": proba})
    if TOP_K > 0:
        idx = out["proba"].nlargest(TOP_K).index
        out["top_k_flag"] = 0
        out.loc[idx, "top_k_flag"] = 1
else:
    pred = pipe.predict(df)
    out = pd.DataFrame({"pred": pred})

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
out.to_csv(args.output, index=False, encoding="utf-8")
print("Saved:", args.output)