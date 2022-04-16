from fastapi.testclient import TestClient

from source.query.main import app

client = TestClient(app)

# a unit test that tests the status code and response of the defined path
def test_get_path():
    r = client.get("/items/42")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 1 of 42"}

# a unit test that tests the status code and response of the defined query
def test_get_path_query():
    r = client.get("/items/42?count=5")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 5 of 42"}

# a unit test that tests the status code
def test_get_malformed():
    r = client.get("/items")
    assert r.status_code != 200