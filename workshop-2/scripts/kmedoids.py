import numpy as np

class CustomKMedoids:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-3):
        """
        Initialize the KMedoids clustering model.

        Parameters:
        - n_clusters (int): Number of clusters to form.
        - max_iter (int): Maximum number of iterations for the KMedoids algorithm.
        - tol (float): Tolerance to declare convergence. If the change in medoids is less than tol, the algorithm stops.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.medoids = None

    def fit(self, X):
        """
        Fit the KMedoids model to the data.

        Parameters:
        - X (numpy.ndarray): Input data, shape (n_samples, n_features).

        Returns:
        - self: The fitted model.
        """
        
        n_samples, n_features = X.shape
        # Initialize medoids randomly
        medoid_indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        self.medoids = X[medoid_indices]

        for _ in range(self.max_iter):
            # Assign each point to the nearest medoid
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2), axis=1)

            # Update medoids by choosing the point that minimizes the sum of distances
            new_medoids = np.empty_like(self.medoids)
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                distances = np.sum(np.abs(cluster_points[:, np.newaxis] - cluster_points), axis=2)
                total_distances = np.sum(distances, axis=1)
                new_medoids[i] = cluster_points[np.argmin(total_distances)]

            # Check for convergence
            if np.all(new_medoids == self.medoids):
                break

            # Check for convergence using tolerance
            if np.sum(np.abs(new_medoids - self.medoids)) < self.tol:
                break

            self.medoids = new_medoids

        return self

    def transform(self, X):
        """
        Assign each data point to the nearest medoid and return cluster labels.

        Parameters:
        - X (numpy.ndarray): Input data, shape (n_samples, n_features).

        Returns:
        - labels (numpy.ndarray): Cluster labels for each data point.
        """
        # Assign each point to the nearest medoid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2), axis=1)
        return labels

    def fit_transform(self, X):
        """
        Fit the KMedoids model to the data and return cluster labels.

        Parameters:
        - X (numpy.ndarray): Input data, shape (n_samples, n_features).

        Returns:
        - labels (numpy.ndarray): Cluster labels for each data point.
        """
        self.fit(X)
        return self.transform(X)