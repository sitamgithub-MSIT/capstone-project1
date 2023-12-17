# Required imports
import pytest
from flask import url_for

# Importing the Flask application
from main import app


@pytest.fixture
def client():
    """
    Fixture function that sets up a test client for the Flask app.

    Returns:
        FlaskClient: The test client object.
    """

    # Setting the app in testing mode
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """
    Test the index route of the application.

    Args:
        client: The test client for making HTTP requests.

    Returns:
        None

    Raises:
        AssertionError: If the response status code is not 200 or if the expected message is not found in the response data.
    """

    # Test GET request and response
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome to Medical Charges Prediction" in response.data


def test_predict_data_route(client):
    """
    Test the '/predict' route of the application.

    Args:
        client: The Flask test client.

    Returns:
        None
    """

    # Test GET request and response
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Medical Charges Prediction" in response.data

    # Test POST request
    response = client.post(
        "/predict",
        data={
            "age": "30",
            "sex": "male",
            "bmi": "25.5",
            "children": "2",
            "smoker": "no",
            "region": "southwest",
        },
    )

    # Asserting the response
    assert response.status_code == 200
    assert b"The prediction is" in response.data
