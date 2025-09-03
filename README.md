# Customer Retention Prediction Service

##  Цель проекта
Клиент (онлайн-сервис с подпиской) хотел снизить отток пользователей.  
Задача  предсказывать вероятность ухода (churn) для каждого клиента, чтобы вовремя запускать удерживающие кампании.

---

##  Решение
- ML-пайплайн: **OneHotEncoder + StandardScaler → Logistic Regression (class_weight="balanced")**
- Учтён дисбаланс классов (~25% churn / 75% non-churn)
- Интерфейсы:
  - **API (FastAPI)** для интеграции в CRM
  - **CLI (predict.py)** для пакетного прогноза CSV
- Порог/Top-K вынесены в config.yaml

---

## 📊 Метрики качества
| Метрика   | Значение |
|-----------|----------|
| ROC-AUC   | 0.840 |
| PR-AUC    | 0.635 |
| F1        | 0.621 |
| Recall    | 0.759 |
| Precision | 0.526 |
| Accuracy  | 0.754 |

 Интерпретация: модель хорошо находит клиентов с риском ухода (высокий Recall), что важно для удержания.

---

##  Использование

### Обучение
~~~bash
python src/train.py --input data/customers.csv --output model/model.pkl
~~~

### Предсказание (CLI)
~~~bash
python predict.py --input new_clients.csv --config config.yaml --output predictions.csv
~~~

predictions.csv:
~~~csv
proba,label
0.82,1
0.15,0
~~~

### API
~~~bash
uvicorn app:app --host 0.0.0.0 --port 8000
~~~

Пример:
~~~bash
curl -X POST "http://127.0.0.1:8000/predict_one" ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":{\"tenure\":12,\"MonthlyCharges\":70.5,\"Contract\":\"Month-to-month\"}}"
~~~

---

##  Конфигурация (config.yaml)
~~~yaml
title: "Customer Churn Prediction"
model_path: "model/model.pkl"
target_col: "Churn"
drop_cols: ["customerID"]
threshold: 0.55
top_k: 1000
~~~

---

##  Структура
customer_retention/
 data/
├── model/
├── src/ (train.py, infer.py, app.py, predict.py)
├── config.yaml
├── report/
└── README.md

## ✅ Результат для заказчика
- Модель с понятными метриками
- API/CLI для интеграции
- Управление порогом и Top-K
- Отчёт с визуализациями