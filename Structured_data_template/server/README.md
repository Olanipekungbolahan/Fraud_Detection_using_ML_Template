Flask prediction service

This directory contains a minimal Flask app to serve saved model packages produced by the training pipeline.

Run (development):

```bash
# from repo root
python Structured_data_template/server/app.py --models-dir Structured_data_template/train/config/../models --host 127.0.0.1 --port 8080
```

Or use gunicorn for production-style run:

```bash
# install requirements
pip install -r requirements_minimal.txt

# run with gunicorn
gunicorn --bind 0.0.0.0:8080 "Structured_data_template.server.app:create_app('./models')"
```

Example request (single row or list):

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"Time": 1, "V1": 0.1, "Amount": 100.0}]}'
```

Response (JSON):

{
  "predictions": ["0"],
  "probabilities": [[0.998, 0.002]],
  "model_file": "random_forest_classification_model.pkl"
}
