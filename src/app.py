import os, io, json, yaml, joblib, pandas as pd
from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
import jsonschema
import json

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

MODEL_PATH = CFG.get("model_path", "model/model.pkl")
TARGET_COL = CFG.get("target_col")          # Р С”Р С•Р В»Р С•Р Р…Р С”Р В° РЎвЂљР В°РЎР‚Р С–Р ВµРЎвЂљР В° (Р ВµРЎРѓР В»Р С‘ Р ВµРЎРѓРЎвЂљРЎРЉ Р Р†Р С• Р Р†РЎвЂ¦Р С•Р Т‘Р Вµ РІР‚вЂќ РЎС“Р Т‘Р В°Р В»Р С‘Р С)
DROP_COLS  = CFG.get("drop_cols", [])       # Р С”Р С•Р В»Р С•Р Р…Р С”Р С‘ Р Т‘Р В»РЎРЏ РЎС“Р Т‘Р В°Р В»Р ВµР Р…Р С‘РЎРЏ
TEXT_COL   = CFG.get("text_col")            # Р Т‘Р В»РЎРЏ РЎвЂљР ВµР С”РЎРѓРЎвЂљР С•Р Р†РЎвЂ№РЎвЂ¦ Р СР С•Р Т‘Р ВµР В»Р ВµР в„– (sentiment)
THRESHOLD  = float(CFG.get("threshold", 0.5))
TOP_K      = int(CFG.get("top_k", 0))

pipe = joblib.load(MODEL_PATH)

# Р—Р°РіСЂСѓР¶Р°РµРј СЃС…РµРјСѓ РІР°Р»РёРґР°С†РёРё, РµСЃР»Рё РµСЃС‚СЊ
# Загружаем схему валидации, если есть (без падения на битом JSON)
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schemas", "input_schema.json")
INPUT_SCHEMA = None
if os.path.exists(SCHEMA_PATH):
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            INPUT_SCHEMA = json.load(f)
    except Exception as e:
        print(f"[schema] warning: cannot load schema from {SCHEMA_PATH}: {e}")
        INPUT_SCHEMA = None

app = FastAPI(title=CFG.get("title", "ML Service"))

class Row(BaseModel):
    features: dict

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = list(DROP_COLS)
    if TARGET_COL and TARGET_COL in df.columns:
        cols_to_drop.append(TARGET_COL)
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    if TEXT_COL:
        # Р ВµРЎРѓР В»Р С‘ Р С—РЎР‚Р С‘РЎРѓР В»Р В°Р В»Р С‘ 1 Р С”Р С•Р В»Р С•Р Р…Р С”РЎС“ Р В±Р ВµР В· Р С‘Р СР ВµР Р…Р С‘  Р Р…Р В°Р В·Р С•Р Р†РЎвЂР С Р ВµРЎвЂ Р С”Р В°Р С” TEXT_COL
        if df.shape[1] == 1 and TEXT_COL not in df.columns:
            df.columns = [TEXT_COL]
    return df

@app.post("/predict_one")
def predict_one(row: Row):
    data = row.features
    if INPUT_SCHEMA:
        jsonschema.validate(instance=data, schema=INPUT_SCHEMA)
    df = pd.DataFrame([row.features])
    if INPUT_SCHEMA:
        for rec in df.to_dict(orient='records'):
            jsonschema.validate(instance=rec, schema=INPUT_SCHEMA)
    df = _prepare_df(df)
    if hasattr(pipe, "predict_proba"):
        proba = float(pipe.predict_proba(df)[:, 1][0])
        return {"proba": proba, "label": int(proba >= THRESHOLD)}
    else:
        pred = float(pipe.predict(df)[0])
        return {"pred": pred}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    if INPUT_SCHEMA:
        for rec in df.to_dict(orient='records'):
            jsonschema.validate(instance=rec, schema=INPUT_SCHEMA)
    df = _prepare_df(df)

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

    out_path = "predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    return {"saved": out_path, "n_rows": len(out)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    resp = await call_next(request)
    print(f"{request.method} {request.url.path} -> {resp.status_code}")
    return resp
