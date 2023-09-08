from flask import Flask, request, jsonify, send_file
import numpy as np

# Create a Flask application
app = Flask(__name__)

# Define a route for custom PCA
@app.route('/custom_pca', methods=['POST'])
def custom_pca():
    try:
        # Receives the dat as JSON format input
        data_x = request.json['data_x']
        data_y = request.json['data_y']

        # Initialize and fit a custom PCA model
        custom_pca_prod = CustomPCA(n_components=2)
        dim_red = custom_pca_prod.fit_transform(np.array(data_x))

        # Create a scatter plot
        plot = plt.scatter(dim_red[:,0], dim_red[:,1], c=data_y)
        plt.legend(handles=plot.legend_elements()[0],
           labels=list(data_y))

        # Save the scatter plot as an image
        plt.savefig('pca.png')

        # Return the dimensionality reduction results as a JSON response
        return jsonify({'dimentionality_reduction': dim_red.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

# Define a route to view the custom PCA plot
@app.route('/view_custom_pca', methods=['GET'])
def view_custom_pca():
    try:
        # Return the PCA plot as a PNG image
        return send_file('pca.png', mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5001)