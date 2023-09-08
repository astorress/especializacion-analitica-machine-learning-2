import numpy as np

class CustomSVD:
    def __init__(self, n_components=2):
        """
        Initialize the SVD object.

        Parameters:
        - n_components (int): The number of components to retain.
        """
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None

    def fit(self, matrix):
        """
        Compute matrix standardization, it is, take a matrix and change it so that its mean is equal to 0
        and variance is 1 and Fit the SVD model to the input data.

        Parameters:
        - Matriz (numpy.ndarray): The input data matrix.
        """
        self.mean = np.mean(matrix, axis=0)
        self.std = np.std(matrix, axis=0)
        self.matrix_standardized = (matrix - self.mean) / self.std

        self.U, self.S, self.Vt = np.linalg.svd(matrix)

    def transform(self, matrix):
        """
        Transform the input data using the SVD components.

        Parameters:
        - matrix (numpy.ndarray): The input data matrix to transform.

        Returns:
        - matrix_reduced (numpy.ndarray): The transformed data with reduced dimensions.
        """
        matrix_reduced = np.dot(matrix, self.Vt.T[:, :self.n_components])
        return matrix_reduced

    def predict(self, matrix):
        """
        Reconstruct the input data using the SVD components.

        Parameters:
        - matrix (numpy.ndarray): The input data matrix to reconstruct.

        Returns:
        - matrix_reconstructed (numpy.ndarray): The reconstructed data matrix.
        """
        matrix_reconstructed = np.dot(self.transform(matrix), self.Vt[:self.n_components, :])
        return matrix_reconstructed


