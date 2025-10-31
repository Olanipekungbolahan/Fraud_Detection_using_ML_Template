import json
import os
import tempfile
import unittest

import joblib

from Structured_data_template.server.app import create_app


class DummyModel:
    def predict(self, X):
        # return zero for each row
        return [0] * len(X)

    def predict_proba(self, X):
        # return two-class probabilities
        return [[0.7, 0.3] for _ in range(len(X))]


class ServerTest(unittest.TestCase):
    def test_predict_with_dummy_model(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "000_dummy_model.pkl")
            package = {
                "model": DummyModel(),
                "preprocessor": None,
                "feature_names": ["A", "B"]
            }
            joblib.dump(package, model_path)

            app = create_app(td)
            client = app.test_client()

            payload = {"rows": [{"A": 1, "B": 2}]}
            resp = client.post("/predict", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            # predictions should be a single value list
            self.assertIn("predictions", data)
            self.assertEqual(data["predictions"], [0])
            # probabilities may be present depending on model return type
            if "probabilities" in data:
                self.assertEqual(len(data["probabilities"]), 1)
            # model file name returned
            self.assertIn("model_file", data)
            self.assertEqual(data["model_file"], os.path.basename(model_path))


if __name__ == "__main__":
    unittest.main()
