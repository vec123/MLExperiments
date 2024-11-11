import numpy as np
import kernel_lib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF,Matern, WhiteKernel, ConstantKernel as C
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF, ConstantKernel as C, WhiteKernel


data = np.load('data/data_noisy_parallel_lines.npy')

x_values = data[:, 0]
y_values = data[:, 1]
x_train  = x_values.reshape(-1, 1)
y_train = y_values.reshape(-1, 1)

rbf_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))#  C(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=6.0, length_scale_bounds=(1e-2, 1e2)) 
        # + RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        # + WhiteKernel(noise_level=0.5)
period = 2 * np.pi      # Sine wave period
length_scale = 2.0      # Controls smoothness
sigma_f = 1.0 
periodic_kernel = sigma_f * ExpSineSquared(length_scale=length_scale, periodicity=period)
exact_sine_wave_kernel = kernel_lib.ExactSineWaveKernel(amplitude=1.0, period=2 * np.pi)
rbf_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))


gp = GaussianProcessRegressor(kernel=rbf_kernel, n_restarts_optimizer=0, alpha=3)
gp.fit(x_train, y_train)

x_test = np.linspace(min(x_train), max(x_train), 500).reshape(-1, 1)
# Make predictions using the GP model
y_pred, sigma = gp.predict(x_test, return_std=True)


from matplotlib import pyplot as plt
plt.figure(figsize=(12, 6))
plt.scatter(x_train, y_train, color='grey', label='Training Data', alpha=0.8, s=100)
plt.plot(x_test, y_pred, 'r-', label='GP Mean Prediction', lw=2)
#plt.scatter(x, sine_values, color='blue', label='Sine Wave Values', alpha=0.6)
plt.fill_between(x_test.flatten(), y_pred - 2*sigma, y_pred + 2*sigma, color='blue', alpha=0.2, label='GP 1-sigma Band')
#plt.fill_between(x_train, y_pred_mean - sigma_pred, y_pred_mean + sigma_pred, color='blue', alpha=0.2, label='Varying Sigma Band')
plt.title('Gaussian Process Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig("images/GP_output.png")


