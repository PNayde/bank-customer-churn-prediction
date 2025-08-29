from starlette.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_dummy():
    r = client.post("/predict", json={"rows":[{"x1":1.0,"x2":0.2},{"x1":-0.9,"x2":0.1}]})
    assert r.status_code == 200
    body = r.json()
    assert "preds" in body and len(body["preds"]) == 2
