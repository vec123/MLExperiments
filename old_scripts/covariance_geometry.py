import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# 1. White Noise
white_noise = np.random.normal(0, 1, size=(100, 2))

# 2. Data 
np.random.seed(42) 
x = np.linspace(-3* np.pi, 3 * np.pi, 500)  # x values

mean_x =   x
variance_x = 0.01
x_samples = np.random.normal(loc=mean_x, scale=np.sqrt(variance_x), size=len(x))

mean_y = np.sin(x)
variance_y = 0.5
y_samples = np.random.normal(loc=mean_y, scale=np.sqrt(variance_y), size=len(x))

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(x_samples, y_samples, label='Sampled Data (y)', alpha=0.7, c="blue")
#plt.scatter(x, y_samples, color='red', label='Sampled Data (y)', alpha=0.7, s=10)
#plt.scatter(x, mean_y + np.sqrt(variance_y), mean_y - np.sqrt(variance_y), color='gray', alpha=0.3, label='Variance Region')
plt.title('Distribution with Sinusoidal Variance for y and Constant Variance for x')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Gaussian_Process_Regression.png")


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF,Matern, WhiteKernel, ConstantKernel as C
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF, ConstantKernel as C, WhiteKernel

x_train  = x.reshape(-1, 1)
y_train = y_samples.reshape(-1, 1)
# Gaussian Process setup with a linear kernel
#kernel = C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2))
#kernel = C(2.0, (1e-3, 1e3)) * RBF(length_scale=5.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=10)
kernel = C(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=2.0, length_scale_bounds=(1e-2, 1e2)) \
         + RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
         + WhiteKernel(noise_level=0.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, alpha=1e-2)

# Fit GP to the training data
gp.fit(x_train, y_train)

# 3. Predict the mean and compute residuals
y_pred_mean, _ = gp.predict(x_train, return_std=True)
residuals = (y_train - y_pred_mean) ** 2
variance_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gp_variance = GaussianProcessRegressor(kernel=variance_kernel, n_restarts_optimizer=10, alpha=1e-4)
gp_variance.fit(x_train, np.log(residuals + 1e-6))  
# 5. Predict the log-variance and compute the sigma
log_variance_pred, _ = gp_variance.predict(x_train, return_std=True)
sigma_pred = np.sqrt(np.exp(log_variance_pred))

# Generate test points for prediction
x_test = np.linspace(min(x_samples), max(x_samples), 500).reshape(-1, 1)
# Make predictions using the GP model
y_pred, sigma = gp.predict(x_test, return_std=True)
# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(x_train, y_train, color='red', label='Training Data', alpha=0.6)
plt.plot(x_test, y_pred, 'b-', label='GP Mean Prediction (Linear Kernel)', lw=2)
#plt.fill_between(x_test.flatten(), y_pred - 3*sigma, y_pred + 3*sigma, color='blue', alpha=0.2, label='GP 1-sigma Band')
plt.fill_between(x_train, y_pred_mean - sigma_pred, y_pred_mean + sigma_pred, color='blue', alpha=0.2, label='Varying Sigma Band')
plt.title('Gaussian Process Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig("Gaussian_Process_Regression_Result.png")
