import numpy as np

class CustomKMeans:
    def __init__(self, n_clusters=8, max_iters=300, tol=1e-4):
        """
        KMeans clustering implementation.

        Parameters:
        - n_clusters: int, number of clusters (default: 8)
        - max_iters: int, maximum number of iterations (default: 300)
        - tol: float, tolerance for convergence (default: 1e-4)
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        """
        Fit the KMeans model to the data.

        Parameters:
        - X: array-like, shape (n_samples, n_features), input data

        Returns:
        - self: fitted KMeans model
        """
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(len(X), self.n_clusters, replace=False)]

        for iter in range(self.max_iters):
            # Calculate distances between data points and centroids
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            
            # Assign each data point to the nearest centroid
            labels = np.argmin(distances, axis=1)
            
            # Calculate new centroids as the mean of data points in each cluster
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # Check for convergence based on the tolerance
            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                break
            
            self.centroids = new_centroids
        
        self.labels_ = labels
        self.inertia_ = np.sum((X - self.centroids[labels]) ** 2)  # Total squared distance
        
        return self

    def fit_transform(self, X):
        """
        Fit the KMeans model to the data and return cluster labels.

        Parameters:
        - X: array-like, shape (n_samples, n_features), input data

        Returns:
        - labels: array-like, shape (n_samples,), cluster labels
        """
        self.fit(X)
        return self.labels_

    def transform(self, X):
        """
        Transform input data and return cluster labels.

        Parameters:
        - X: array-like, shape (n_samples, n_features), input data

        Returns:
        - labels: array-like, shape (n_samples,), cluster labels
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)