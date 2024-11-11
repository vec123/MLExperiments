

from sklearn.gaussian_process.kernels import Kernel
from scipy.linalg import cholesky, LinAlgError,eigvalsh


class SineWaveKernel(Kernel):
    def __init__(self, amplitude=1.0, period=2 * np.pi):
        self.amplitude = amplitude
        self.period = period

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)

        # Calculate pairwise distance matrix
        dists = np.subtract.outer(X[:, 0], Y[:, 0])
        K = 10 + 1* self.amplitude * np.sin(dists * (2 * np.pi / self.period)) ** 2
 
        # Add small value to diagonal to enforce positive definiteness
        if X.shape[0] == Y.shape[0] and np.allclose(X, Y):
            K[np.diag_indices_from(K)] += 1e-6  # Add jitter to the diagonal


        if X.shape[0] == Y.shape[0] and np.allclose(X, Y):
            # Compute eigenvalues of K
            eigenvalues = eigvalsh(K)
            print("Eigenvalues before adjustment:", eigenvalues)

            # If any eigenvalues are non-positive, add jitter
            if np.any(eigenvalues <= 0):
                alpha = 1e-6
                K += alpha * np.eye(K.shape[0])
                print("Adjusted matrix K with alpha added to diagonal.")
                print("Adjusted K:", K)
                eigenvalues_adjusted = eigvalsh(K)
                print("Eigenvalues after adjustment:", eigenvalues_adjusted)

        if eval_gradient:
            grad_amplitude = K / self.amplitude
            grad_period = (4 * np.pi * dists * K) / (self.period ** 2)
            return K, np.stack([grad_amplitude, grad_period], axis=2)
                # Check if K is positive definite
                
        print("K", K)
        return K


    def diag(self, X):
        return np.full(X.shape[0], self.amplitude)

    def is_stationary(self):
        return True
    
class ExactSineWaveKernel(Kernel):
    def __init__(self, amplitude=1.0, period=2 * np.pi, phase=0.0):
        self.amplitude = amplitude
        self.period = period
        self.phase = phase

    def __call__(self, X, Y=None, eval_gradient=False):
        # Map each input to the known sine wave's value
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)

        # Compute sine wave values for X and Y based on known sine wave parameters
        sin_X = self.amplitude * np.sin((X[:, 0] / self.period) * (2 * np.pi) + self.phase)
        sin_Y = self.amplitude * np.sin((Y[:, 0] / self.period) * (2 * np.pi) + self.phase)

        # Kernel as an outer product of sine values, producing deterministic similarity
        K = np.outer(sin_X, sin_Y)

        if eval_gradient:
            return K, np.zeros((X.shape[0], Y.shape[0], 0))
        
        return K

    def diag(self, X):
        # Return the amplitude for each input's sine wave value
        return np.full(X.shape[0], self.amplitude**2)

    def is_stationary(self):
        return True
