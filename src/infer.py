import os, sys, argparse, joblib, pandas as pd

# доступ к корню, если потребуется
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/model.pkl")
    parser.add_argument("--input", required=True, help="CSV без столбца целевой (или с ним — будет проигнорирован)")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    pipe = joblib.load(args.model)
    df = pd.read_csv(args.input)

    # если есть столбец Churn — не используем его в прогнозе
    for col in ["Churn", "target", "label"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    proba = pipe.predict_proba(df)[:, 1]
    out = pd.DataFrame({"proba": proba})
    out.to_csv(args.output, index=False)
    print("Saved:", args.output)

if __name__ == "__main__":
    main()