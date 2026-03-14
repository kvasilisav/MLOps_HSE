import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import _reset_db_and_storage


@pytest.fixture
def client():
    _reset_db_and_storage()
    return TestClient(app)


@pytest.fixture
def model_file():
    return ("model.pkl", io.BytesIO(b"fake_model_weights_123"))


def test_register_model(client, model_file):
    response = client.post(
        "/models",
        data={
            "name": "test_model",
            "team": "mlds_1",
            "metadata": '{"dataset": "train_v1"}',
            "tags": '{"task": "classification"}',
            "status": "staging",
        },
        files={"file": model_file},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "mlds_1_test_model"
    assert data["version"] == 1
    assert "path" in data


def test_list_models_empty(client):
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == []


def test_list_models_after_register(client, model_file):
    client.post("/models", data={"name": "my_model", "team": "mlds_1"}, files={"file": model_file})
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 1
    assert any(m["name"] == "my_model" for m in models)


def test_list_models_filter_by_team(client, model_file):
    client.post("/models", data={"name": "a", "team": "team_x"}, files={"file": model_file})
    client.post("/models", data={"name": "b", "team": "team_y"}, files={"file": model_file})
    response = client.get("/models?team=team_x")
    assert response.status_code == 200
    models = response.json()
    assert all(m["team"] == "team_x" for m in models)


def test_get_model_details(client, model_file):
    client.post("/models", data={"name": "detail_test", "team": "mlds_1"}, files={"file": model_file})
    response = client.get("/models/mlds_1_detail_test")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "mlds_1_detail_test"
    assert len(data["versions"]) >= 1
    assert data["versions"][0]["version"] == 1


def test_get_model_not_found(client):
    response = client.get("/models/nonexistent_model_xyz")
    assert response.status_code == 404


def test_download_version(client, model_file):
    client.post("/models", data={"name": "download_test", "team": "mlds_1"}, files={"file": model_file})
    response = client.get("/models/mlds_1_download_test/versions/1")
    assert response.status_code == 200
    assert response.content == b"fake_model_weights_123"


def test_download_version_not_found(client):
    response = client.get("/models/mlds_1_xxx/versions/999")
    assert response.status_code == 404


def test_update_version_status(client, model_file):
    client.post("/models", data={"name": "update_test", "team": "mlds_1"}, files={"file": model_file})
    response = client.patch(
        "/models/mlds_1_update_test/versions/1",
        json={"status": "production"},
    )
    assert response.status_code == 200
    resp = client.get("/models/mlds_1_update_test")
    assert resp.json()["versions"][0]["status"] == "production"


def test_versioning(client, model_file):
    client.post(
        "/models",
        data={"name": "versioned", "team": "mlds_1"},
        files={"file": ("v1.pkl", io.BytesIO(b"v1"))},
    )
    response = client.post(
        "/models",
        data={"name": "versioned", "team": "mlds_1"},
        files={"file": ("v2.pkl", io.BytesIO(b"v2"))},
    )
    assert response.json()["version"] == 2
    resp = client.get("/models/mlds_1_versioned/versions/2")
    assert resp.content == b"v2"
