from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_empty_message_is_rejected():
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_approve_unknown_thread_is_404():
    response = client.post("/approve", json={"thread_id": "does-not-exist", "approve": True})
    assert response.status_code == 404