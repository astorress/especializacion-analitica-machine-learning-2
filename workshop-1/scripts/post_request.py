# Import necessary libraries
from sklearn.datasets import load_digits
import requests

# Load the digits dataset from scikit-learn
mnist = load_digits(return_X_y=False)

# Extract features (X) and target labels (y) from the dataset
x_mnist = mnist.data
y_minst = mnist.target

# URL of the server to which you want to send the POST request
url = 'http://192.168.1.2:5001/custom_pca'

# Data you want to send in the body of the POST request (in JSON format)
data = {
    'data_x': x_mnist.tolist(),
    'data_y': y_minst.tolist()
}

# Perform a POST request to the specified URL with the data
response = requests.post(url, json=data)

# Checks if the request was successful (HTTP status code 200)
if response.status_code == 200:
    # Process the server's response (assuming it returns dimensionality reduction results)
    response_data = response.json()
    display(response_data['dimentionality_reduction']) # Display the results
else:
    print('Error in the request:', response.status_code, response.text)