import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# --------- White Noise (Identity Covariance Matrix)
zero_mean = [0, 0]
cov_matrix_white_noise = [[1, 0], [0, 1]]
data_white_noise = np.random.multivariate_normal(zero_mean, cov_matrix_white_noise, size=500)
data = np.array(data_white_noise, dtype=np.float32)
np.save('data/data_white_noise.npy', data)
# Saving the data to a file

# ------------- Data with a specified constant Covariance Matrix
cov_matrix_specific = [[3, 1], [1, 2]]
data_cov = np.random.multivariate_normal(zero_mean, cov_matrix_specific, size=500)
data = np.array(data_cov, dtype=np.float32)
np.save('data/data_cov.npy', data)


# ------------- Data with a specified constant Covariance Matrix
cov_matrix_specific = [[3, 1], [1, 2]]
data_white_noise = np.random.multivariate_normal(zero_mean, cov_matrix_white_noise, size=500)
mean = np.mean(data_white_noise, axis=0)
data_cov = np.random.multivariate_normal(mean, cov_matrix_specific, size=500)
data = np.array(data_cov, dtype=np.float32)
np.save('data/data_cov.npy', data)

# -------------- Data Distributed Around a parabola with constant Nois
x_values_parabola = np.linspace(-3, 3, 5000)
parabola_values = x_values_parabola**2
data_parabola = np.vstack((x_values_parabola, parabola_values)).T
cov_matrix_parabola_noise = [[0.5, 0.2], [0.2, 0.3]]  # Covariance matrix for noise
noise_parabola = np.random.multivariate_normal(zero_mean, cov_matrix_parabola_noise, size=5000)
data_parabola += noise_parabola
data = np.array(data_parabola, dtype=np.float32)
np.save('data/data_parabola.npy', data)

# -------------- Data Distributed Around a Circle with constant noise
number_of_points = 5000
theta = np.linspace(0, 2 * np.pi, number_of_points)
radius = 5
circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)
data_circle = np.vstack((circle_x, circle_y)).T
data_circle_noise = data_circle
cov_matrix_circle_noise = [[1, 0.1], [1, 0.1]]  # Covariance matrix for noise
noise_circle = np.random.multivariate_normal(zero_mean, cov_matrix_circle_noise, size=number_of_points)
data_circle_noise += noise_circle
data = np.array(data_circle_noise, dtype=np.float32)
np.save('data/data_circle_noise.npy', data)


# ---------------------Data Distributed Along a Diagonal Line
cov_matrix_diagonal = [[1, 0.9], [0.9, 1]]
data_diagonal = np.random.multivariate_normal(mean, cov_matrix_diagonal, size=500)
data = np.array(data_diagonal, dtype=np.float32)
np.save('data/data_diagonal.npy', data)

#------------------------Data Distributed Along a y axis 
# Generate uniform data along the y-axis
y_values = np.random.uniform(-100, 100, 100)
x_values = np.random.normal(0, np.sqrt(5), 100)  # Standard deviation is the square root of variance (sqrt(5))
#x_values = x_values *0
data_uniform_line = np.vstack((x_values, y_values)).T
data = np.array(data_uniform_line, dtype=np.float32)
np.save('data/data_uniform_line.npy', data)

# ---------------------Data Distributed Along a digaonal rotated  
phi = np.pi / 4  # 45 degrees
transformation_matrix = np.array([
    [np.cos(phi), np.sin(phi)],
    [-np.sin(phi), np.cos(phi)]
])
data_uniform_line_tf = data_uniform_line @ transformation_matrix.T
original_cov_matrix = np.cov(data_uniform_line, rowvar=False)
transformed_cov_matrix = transformation_matrix @ original_cov_matrix @ transformation_matrix.T

data = np.array(data_uniform_line_tf, dtype=np.float32)
np.save('data/data_uniform_line_tf.npy', data)


#-------------------------Data Distributed Along a y axis 
y_values = np.random.uniform(-100, 100, 500)
y_values.sort() 
mid_point = len(y_values) // 2
y_values_first_half = y_values[:mid_point]
y_values_second_half = y_values[mid_point:]
cov_matrix_first_half = [[5, 0], [0, 0.1]]  # Covariance matrix for the first half
noise_first_half = np.random.multivariate_normal(zero_mean, cov_matrix_first_half, size=mid_point)
cov_matrix_second_half = [[100, 0], [0, 0.1]]  # Covariance matrix for the second half
noise_second_half = np.random.multivariate_normal(zero_mean, cov_matrix_second_half, size=len(y_values) - mid_point)
x_values_first_half = noise_first_half[:, 0]
x_values_second_half = noise_second_half[:, 0]
x_values = np.concatenate([x_values_first_half, x_values_second_half])
y_values = np.concatenate([y_values_first_half, y_values_second_half])
data_combined = np.vstack((x_values, y_values)).T


# -----------------Data Distributed Along a y axis sine-noise
# Generate y-values uniformly along the y-axis
y_values = np.random.uniform(-100, 100, 5000)
y_values.sort()  
# Initialize arrays to store x-values
x_values = np.zeros_like(y_values)
noise_values = []
for i, y in enumerate(y_values):
    # Define a continuously varying covariance for x based on y
    #variance_x = 5 + (y + 100) * 0.95  # Map y in [-100, 100] to a variance in [5, 195]
    variance_x = 5 + 300 * (0.5 + 0.5 * np.sin(3*y * np.pi /50))  # Scales sine wave to [5, 50]
    covariance_matrix = [[variance_x, 0], [0, 0.1]]  # Keep variance for y small
    noise = np.random.multivariate_normal(zero_mean, covariance_matrix)
    noise_values.append(noise)
    x_values[i] = noise[0]
    
data_continuous_covariance = np.vstack((x_values, y_values)).T
data = np.array(data_continuous_covariance, dtype=np.float32)
np.save('data/data_continuous_covariance.npy', data)


#------------------------- Map to circle with sine-noise
theta = (y_values - y_values.min()) / (y_values.max() - y_values.min()) * 2 * np.pi  # Map y in [-100, 100] to [0, 2*pi]
print("len theta:", len(theta))
radius = 200
circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)
noise_circle = np.zeros((len(theta), 2))
for i in range(len(theta)):
    # Determine the normal direction at this point (perpendicular to the tangent)
    normal_direction = np.array([np.cos(theta[i]), np.sin(theta[i])])
    noise_circle[i] = noise_values[i][0] * normal_direction 
data_circle_sine_variance = np.vstack((circle_x, circle_y)).T + noise_circle

data = np.array(data_circle_sine_variance, dtype=np.float32)
np.save('data/data_circle_sine_variance.npy', data)

#---------------------- Map to a parabola with sine-noise
t = (y_values - y_values.min()) / (y_values.max() - y_values.min()) * 2 - 1  # Map y to range [-1, 1]
a = 1 # Controls the width of the parabola
parabola_x = t
parabola_y = a * t**2
noise_parabola = np.zeros((len(t), 2))
for i in range(len(t)):
    # The tangent vector at this point is [1, 2*a*t[i]]
    tangent = np.array([1, 2 * a * t[i]])
    # The normal vector is perpendicular to the tangent
    normal_direction = np.array([-tangent[1], tangent[0]])
    normal_direction /= np.linalg.norm(normal_direction)  # Normalize
    noise_parabola[i] = noise_values[i][0] * normal_direction
data_parabola_sine_variance = np.vstack((parabola_x, parabola_y)).T +noise_parabola*0.01
data = np.array(data_parabola_sine_variance, dtype=np.float32)
np.save('data/data_parabola_const_variance.npy', data)


#--------------Sine wave with constant noise
size = 2000
x_values = np.random.uniform(-10, 10, size)
x_values.sort()  
# Initialize arrays to store x-values
y_values = 3*np.sin(x_values)
cov_matrix_diagonal = [[1, 0], [0, 1]]
noise = np.random.multivariate_normal(zero_mean, cov_matrix_diagonal, size=size)
data_sine_wave = np.vstack((x_values, y_values)).T +noise*0.7
data = np.array(data_sine_wave, dtype=np.float32)
np.save('data/data_sine_const_variance.npy', data)



#--------------------star with three straight lines
num_points = 500
line_length = 10  # Length of each line from the origin
noise_level = 1  # 2D noise level for each line

line_1_x = np.linspace(0, line_length, num_points)
line_1_y = line_1_x
line_1 = np.vstack((line_1_x, line_1_y)).T + np.random.normal(0, noise_level, (num_points, 2))  # Add 2D noise

# Line 2: Along the negative diagonal (-45 degrees)
line_2_x = np.linspace(0, line_length, num_points)
line_2_y = -line_2_x
line_2 = np.vstack((line_2_x, line_2_y)).T + np.random.normal(0, noise_level, (num_points, 2))  # Add 2D noise

# Line 3: Vertically upwards
line_3_x = -1*np.linspace(0, line_length, num_points)
line_3_y = 0.2*np.linspace(0, line_length, num_points)
line_3 = np.vstack((line_3_x, line_3_y)).T + np.random.normal(0, noise_level, (num_points, 2))  # Add 2D noise

# Combine the three lines and cast to float32
data_star_pattern = np.vstack((line_1, line_2, line_3)).astype(np.float32)

# Save the data to a file
np.save('data/data_diagonal_star.npy', data_star_pattern)

#------------------curved star
num_points = 500
line_length = 10  # Length of each line from the origin
noise_level = 1 # 2D noise level for each line
# Generate curved lines with 2D noise
# Line 1: Along a curved positive diagonal (45 degrees, with slight curve added)
line_1_x = np.linspace(0, line_length, num_points)
line_1_y = (0.02*line_1_x**2) *10 # Introduce a slight curve by exponentiating
line_1 = np.vstack((line_1_x, line_1_y)).T + np.random.normal(0, noise_level, (num_points, 2))  # Add 2D noise
# Line 2: Along a curved negative diagonal (-45 degrees, with slight curve added)
line_2_x = np.linspace(0, line_length, num_points)
line_2_y = -(0.01*line_2_x**2) * 7   # Curve in the opposite direction
line_2 = np.vstack((line_2_x, line_2_y)).T + np.random.normal(0, noise_level, (num_points, 2))  # Add 2D noise
# Line 3: Vertically upwards with a curve (curving towards left)
line_3_x = -np.linspace(0, line_length, num_points)
line_3_y = -(0.01*line_2_x**2) * 7 #np.sin(0.8*line_3_x) * 2  # Curve slightly to create an arc upwards, maintaining direction
line_3 = np.vstack((line_3_x, line_3_y)).T + np.random.normal(0, noise_level, (num_points, 2))  # Add 2D noise

# Combine the three curved lines and cast to float32
data_star_pattern_curved = np.vstack((line_1, line_2, line_3)).astype(np.float32)
# Save the data to a file
np.save('data/data_diagonal_curved_star.npy', data_star_pattern_curved)



#----------------------noisy parallel lines
num_points = 500
line_length = 10  # Length of each line from the origin
noise_level = 0.7  # 2D noise level for each line
line_offset = 7  # Distance between the parallel lines

# Generate points along the first line with 2D noise
line_1_x = np.linspace(0, line_length, num_points)
line_1_y = np.zeros(num_points)  # y = 0 for a horizontal line
line_1 = np.vstack((line_1_x, line_1_y)).T + np.random.normal(0, noise_level, (num_points, 2))  # Add 2D noise

# Generate points along the second parallel line with 2D noise, offset by `line_offset`
line_2_x = np.linspace(0, line_length, num_points)
line_2_y = np.ones(num_points) * line_offset  # y = offset for the second horizontal line
line_2 = np.vstack((line_2_x, line_2_y)).T + np.random.normal(0, noise_level, (num_points, 2))  # Add 2D noise

# Combine the two parallel lines and cast to float32
data_parallel_lines = np.vstack((line_1, line_2)).astype(np.float32)
# Save the data to a file
np.save('data/data_noisy_parallel_lines.npy', data_parallel_lines)


#---------------- spirals
def make_spiral(number_of_turns):
    num_points = 1000  # Number of points per spiral
    num_turns = number_of_turns  # Number of turns in each spiral
    noise_level = 3 # 2D noise level for each spiral
    theta_offset = np.pi  # Offset angle between each spiral
    # Generate three spirals with noise
    spirals = []
    for i in range(2):
        theta = np.linspace(0, num_turns * 2 * np.pi, num_points) + i * theta_offset  # Spiraling outwards
        r = np.linspace(0, 10, num_points)  # Radius increasing outward
        
        # Convert polar coordinates to Cartesian for the spiral pattern
        x = 5*r * np.cos(theta)
        y = 5*r * np.sin(theta)
        
        # Stack the coordinates and add noise
        spiral = np.vstack((x, y)).T + np.random.normal(0, noise_level, (num_points, 2))
        spirals.append(spiral)

    # Combine the three spirals and cast to float32
    data_spirals = np.vstack(spirals).astype(np.float32)
    # Save the data to a file
    np.save('data/data_spirals_{}_turn.npy'.format(number_of_turns), data_spirals)
    return data_spirals

spiral_1 = make_spiral(1)
spiral_15 = make_spiral(1.5)
spiral_2 = make_spiral(2)
spiral_3 = make_spiral(3)


#---------------- concentric circles
def generate_single_circle(n_samples, radius, noise=0.1):
    angles = np.linspace(0, 2 * np.pi, n_samples)
    x = radius * np.cos(angles) + np.random.normal(scale=noise, size=angles.shape)
    y = radius * np.sin(angles) + np.random.normal(scale=noise, size=angles.shape)
    return np.vstack([x, y]).T

# Generate data for concentric circles
def generate_concentric_circles(n_samples, radii, noise=1):
    data = []
    for radius in radii:
        angles = np.linspace(0, 2 * np.pi, n_samples // len(radii))
        x = radius * np.cos(angles) + np.random.normal(scale=noise, size=angles.shape)
        y = radius * np.sin(angles) + np.random.normal(scale=noise, size=angles.shape)
        data.extend(np.vstack([x, y]).T)
    return np.array(data, dtype=np.float32)

n_samples = 1000
single_radius = 3  # Radius for the single circle
radii = [3,15]  # Radii for the concentric circles

# Generate the data
single_circle = generate_single_circle(n_samples, single_radius)
concentric_circles = generate_concentric_circles(n_samples, radii)

# Plot settings
#number_of_figures = 8
fig_size = (8, 8)

plt.figure(figsize=fig_size)
plt.scatter(data_white_noise[:, 0], data_white_noise[:, 1], alpha=0.6, edgecolor='k')
plt.title('White Noise (Identity Covariance)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images\white_noise.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_cov[:, 0], data_cov[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data with Specific Covariance')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/specific_covariance.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_parabola[:, 0], data_parabola[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data Around a Parabola')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/parabola.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_circle_noise[:, 0], data_circle_noise[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data Around a Circle')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/circle.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_diagonal[:, 0], data_diagonal[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data Around a Diagonal')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/diagonal.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_uniform_line[:, 0], data_uniform_line[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data Uniform Line (regular distribution)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/uniform_line (regular distribution).png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_uniform_line_tf[:, 0], data_uniform_line_tf[:, 1], alpha=0.6, edgecolor='k')
plt.title('Transformed Uniform Line  (regular distribution)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/transformed_uniform_line (regular distribution).png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_combined[:, 0], data_combined[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data with Two Variances')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/two_variances.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_continuous_covariance[:, 0], data_continuous_covariance[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data with continously varying variance (sine)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/continous_varying_variance_sine.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_circle_sine_variance[:, 0], data_circle_sine_variance[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data with circle mean and sine variance')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/circle_mean_sine_variance.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_parabola_sine_variance[:, 0], data_parabola_sine_variance[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data with parabola mean and sine variance')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/parabola_mean_sine_variance.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_star_pattern [:, 0], data_star_pattern [:, 1], alpha=0.6, edgecolor='k')
plt.title('Data in a star pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/star_pattern.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_star_pattern_curved [:, 0], data_star_pattern_curved [:, 1], alpha=0.6, edgecolor='k')
plt.title('Data in a curved star pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/curved_star_pattern.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_parallel_lines [:, 0], data_parallel_lines [:, 1], alpha=0.6, edgecolor='k')
plt.title('Data in a curved star pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/curved_star_pattern.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(spiral_1 [:, 0], spiral_1 [:, 1], alpha=0.6, edgecolor='k')
plt.title('Data in a spiral pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/spiral_1_pattern.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(spiral_15 [:, 0], spiral_15 [:, 1], alpha=0.6, edgecolor='k')
plt.title('Data in a spiral pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/spiral_1.5_pattern.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(spiral_2 [:, 0], spiral_2 [:, 1], alpha=0.6, edgecolor='k')
plt.title('Data in a spiral pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/spiral_2_pattern.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(spiral_3 [:, 0], spiral_3 [:, 1], alpha=0.6, edgecolor='k')
plt.title('Data in a spiral pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/spiral_3_pattern.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_sine_wave [:, 0], data_sine_wave [:, 1], alpha=0.6, edgecolor='k')
plt.title('Data in a sine-wave pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/sine_wave_pattern.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(data_sine_wave [:, 0], data_sine_wave [:, 1], alpha=0.6, edgecolor='k')
plt.title('Data in a sine-wave pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/sine_wave_pattern.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(single_circle [:, 0], single_circle [:, 1], alpha=0.6, edgecolor='k')
plt.title('Data in a circular pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/single_circle_pattern.png')
plt.close()

plt.figure(figsize=fig_size)
plt.scatter(concentric_circles [:, 0], concentric_circles [:, 1], alpha=0.6, edgecolor='k')
plt.title('Data in a circular pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('data_images/concentric_circle_pattern.png')
plt.close()

print("---end----")