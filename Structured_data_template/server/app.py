from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from flask import Flask, jsonify, request

LOG = logging.getLogger(__name__)


def find_latest_model(models_dir: Path) -> Path | None:
    if not models_dir.exists():
        return None
    candidates = sorted(models_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_model_package(model_path: Path) -> Dict[str, Any]:
    package = joblib.load(model_path)
    return package


def prepare_input(df: pd.DataFrame, package: Dict[str, Any]) -> pd.DataFrame:
    # If the package contains a preprocessor that expects the original feature names,
    # we pass the DataFrame through it. Otherwise, return df as-is and rely on model.
    preprocessor = package.get("preprocessor")
    if preprocessor is not None:
        try:
            return pd.DataFrame(preprocessor.transform(df), columns=getattr(package.get('model'), '_feature_names', df.columns))
        except Exception:
            # Some preprocessors (ColumnTransformer) need fit/transform API; try transform only
            return pd.DataFrame(preprocessor.transform(df))
    return df


def build_response(preds: List[Any], probs: List[List[float]] | None, package: Dict[str, Any]) -> Dict[str, Any]:
    label_encoder = package.get('label_encoder')
    if label_encoder is not None:
        decoded = label_encoder.inverse_transform(preds)
        preds_out = [str(p) for p in decoded]
    else:
        preds_out = [int(p) if hasattr(p, '__int__') else p for p in preds]

    response = {"predictions": preds_out}
    if probs is not None:
        response["probabilities"] = probs
    return response


def create_app(models_dir: str | Path = "./models") -> Flask:
    app = Flask(__name__)

    models_path = Path(models_dir)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        # Accept either a single JSON object or a list under `rows` key
        payload = request.get_json(force=True)
        if payload is None:
            return jsonify({"error": "empty request body"}), 400

        # Normalize input to list of rows
        if isinstance(payload, dict) and "rows" in payload:
            rows = payload["rows"]
        elif isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            return jsonify({"error": "invalid payload; expected object, list or {rows: [...] }"}), 400

        if len(rows) == 0:
            return jsonify({"error": "no rows provided"}), 400

        # Find and load latest model package
        latest = find_latest_model(models_path)
        if latest is None:
            return jsonify({"error": f"no model files (.pkl) found in {models_path}"}), 500

        try:
            package = load_model_package(latest)
        except Exception as e:
            LOG.exception("Failed loading model package")
            return jsonify({"error": f"failed to load model: {e}"}), 500

        # Convert rows to DataFrame
        try:
            df = pd.DataFrame(rows)
        except Exception as e:
            return jsonify({"error": f"invalid rows format: {e}"}), 400

        # Ensure columns ordering if feature names exist
        feature_names = package.get('feature_names') or getattr(package.get('model'), '_feature_names', None)
        if feature_names is not None:
            # keep only expected columns and ensure order
            df = df.reindex(columns=feature_names)

        # Prepare input using preprocessor if available
        try:
            X = prepare_input(df, package)
        except Exception as e:
            LOG.exception("Failed to prepare input")
            return jsonify({"error": f"failed to prepare input: {e}"}), 500

        model = package.get('model')
        if model is None:
            return jsonify({"error": "model object missing in package"}), 500

        # Predict
        try:
            preds = model.predict(X)
        except Exception as e:
            LOG.exception("Prediction failed")
            return jsonify({"error": f"prediction failed: {e}"}), 500

        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs_arr = model.predict_proba(X)
                # convert numpy arrays to native lists
                probs = probs_arr.tolist()
            except Exception:
                probs = None

        resp = build_response(list(preds), probs, package)
        resp["model_file"] = str(latest.name)

        return jsonify(resp)

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Serve model for predictions via Flask")
    parser.add_argument("--models-dir", default="./models", help="Directory containing model .pkl files")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args()

    app = create_app(args.models_dir)
    app.run(host=args.host, port=args.port)
