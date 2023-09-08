import numpy as np

class CustomTSNE:
  def __init__(self, n_components=2, perplexity=30, learning_rate=0.1, n_iter=200):
      """
      Initialize the t-SNE object.

      Parameters:
      - n_components (int): The number of dimensions in the embedded space.
      - perplexity (float): The perplexity parameter (trade-off between preserving global and local structure).
      - learning_rate (float): The learning rate for gradient descent optimization.
      - n_iter (int): The number of iterations for optimization.
      """
      self.n_components = n_components
      self.perplexity = perplexity
      self.learning_rate = learning_rate
      self.n_iter = n_iter
      self.embedding = None

  def fit_transform(self, X):
      """
      Fit the t-SNE model to the input data and return the embedded data.

      Parameters:
      - X (numpy.ndarray): The input data matrix to perform t-SNE on.

      Returns:
      - X_embedded (numpy.ndarray): The embedded data in the low-dimensional space.
      """
      n_samples, n_features = X.shape
      pairwise_distances = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
      P = self._compute_pairwise_probabilities(pairwise_distances)
      Y = np.random.randn(n_samples, self.n_components)
      Y = self._gradient_descent(Y, P)
      self.embedding = Y
      return Y

  def _compute_pairwise_probabilities(self, pairwise_distances):
      P = np.zeros((pairwise_distances.shape[0], pairwise_distances.shape[0]))
      perplexity_values = [self.perplexity]
      for i in range(pairwise_distances.shape[0]):
          P[i, :] = self._binary_search_perplexity(pairwise_distances[i, :], perplexity_values)
      P = (P + P.T) / (2 * pairwise_distances.shape[0])
      P = np.maximum(P, 1e-12)
      return P

  def _binary_search_perplexity(self, distances_row, perplexity_values, tol=1e-5):
      beta = np.ones(distances_row.shape[0])
      target_entropy = np.log(self.perplexity)
      for _ in range(50):
          current_perplexity = np.sum(np.exp(-distances_row * beta))
          current_entropy = np.log(current_perplexity)
          entropy_diff = current_entropy - target_entropy
          if np.abs(entropy_diff) < tol:
              return beta
          gradient = -distances_row * np.exp(-distances_row * beta)
          gradient_sum = np.sum(gradient)
          beta += (entropy_diff / gradient_sum) * gradient
          beta = np.maximum(beta, 1e-12)
      return beta

  def _gradient_descent(self, Y, P):
      for _ in range(self.n_iter):
          Q = self._compute_low_dimensional_probabilities(Y)
          grad = self._compute_gradient(P, Q, Y)
          Y -= self.learning_rate * grad
      return Y

  def _compute_low_dimensional_probabilities(self, Y):
      pairwise_distances = np.linalg.norm(Y[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)
      Q = 1 / (1 + pairwise_distances ** 2)
      Q /= np.sum(Q)
      return np.maximum(Q, 1e-12)

  def _compute_gradient(self, P, Q, Y):
      gradient = np.zeros(Y.shape)
      for i in range(Y.shape[0]):
          diff = (P[i, :] - Q[i, :])[:, np.newaxis]
          gradient[i, :] = np.sum(diff * (Y[i, :] - Y), axis=0)
      return gradient