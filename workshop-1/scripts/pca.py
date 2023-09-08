import numpy as np

class CustomPCA:
    def __init__(self, n_components=None):
        """
        Initialize the CustomPCA object with the number of components to return.

        Parameters:
        n_components (int or None): Number of principal components to return. If None, all components will be return.
        """
        self.n_components = n_components
        self.mean = None
        self.std_dev = None
        self.components = None

    def fit(self, A):
        """
        Fit the PCA model to the input data A.

        Parameters:
        A (numpy.ndarray): Input data matrix of shape (m, n) where m is the number of rows and n is the number of columns.

        Returns:
        self
        """

        # Standardize the data
        self.mean = np.mean(A, axis=0)
        self.std_dev = np.std(A, axis=0)

        A_std = (A - self.mean) # / self.std_dev

        # Calculate the covariance matrix
        cov_matrix = np.cov(A_std, rowvar=False)

        # Calculate eigenvalue-eigenvector of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the desired number of components
        if self.n_components is not None:
            self.components = eigenvectors[:, :self.n_components]
        else:
            self.components = eigenvectors

        return self

    def fit_transform(self, A):
        """
        Fit the PCA model to the input data A and transform it.

        Parameters:
        A (numpy.ndarray): Input data matrix of shape (m, n) where m is the number of rows and n is the number of columns.

        Returns:
        transformed_data (numpy.ndarray): Transformed data matrix of shape (m, k) where k is the number of return components.
        """
        self.fit(A)
        return self.transform(A)

    def transform(self, A):
        """
        Transform the input data A using the fitted PCA model.

        Parameters:
        A (numpy.ndarray): Input data matrix of shape (m, n) where m is the number of rows and n is the number of columns.

        Returns:
        transformed_data (numpy.ndarray): Transformed data matrix of shape (m, k) where k is the number of return components.
        """
        A_std = (A - self.mean)
        transformed_data = A_std @ self.components
        return transformed_data

