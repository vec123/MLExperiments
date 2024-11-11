import matplotlib.pyplot as plt
import numpy as np

# Define parameters for the circles
radii = [1, 2]
theta = np.linspace(0, 2 * np.pi, 100)

# Create 2D plot of concentric circles
fig_2d, ax_2d = plt.subplots()
circles={ }
samples= {}
for radius in radii:
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    sampled_i = np.random.choice(range(0, len(x)), 30)
    sampled_x = x[sampled_i]
    sampled_y = y[sampled_i]

    ax_2d.plot(x, y, label=f"Radius = {radius}")
    ax_2d.scatter(sampled_x, sampled_y, color='red')



ax_2d.set_aspect('equal')
ax_2d.set_title("2D Concentric Circles")
ax_2d.legend()
plt.savefig('../data_images/concentric_circles_2D.png')

# Create 3D plot of concentric circles with height as radius
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')
for radius in radii:
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(theta, radius)

    
    sampled_i = np.random.choice(range(0, len(x)), 30)
    sampled_x = x[sampled_i]
    sampled_y = y[sampled_i]
    sampled_z = z[sampled_i]

    ax_3d.plot(x, y, z, label=f"Height = {radius}")
    ax_3d.scatter(sampled_x, sampled_y, sampled_z, color='red')

plane_x, plane_y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
plane_z = np.full_like(plane_x, 1.5)  # Plane at height 1.5
ax_3d.plot_surface(plane_x, plane_y, plane_z, color='cyan', alpha=0.5)


ax_3d.set_title("3D Concentric Circles")
ax_3d.legend()
plt.savefig('../data_images/concentric_circles_3D.png')



X2 = np.array([x, y])
X3 = np.array([x, y, z])
print("X2: ", X2.shape)
def get_eigenspace(X):
    eigenval, eigenvec = np.linalg.eig(X)
    space = [eigenval, eigenvec]
    return space

num_samples = X2.shape[0] 
XXT2 = X2@X2.T/num_samples
zero_mean = np.zeros(XXT2.shape[0]) 
samples_XXT2 = np.random.multivariate_normal(zero_mean, XXT2, 200).T
XTX2 = X2.T@X2/num_samples
print("XXT2: ", XXT2.shape)
print("XXT2: ", XXT2.shape)

XXT3 = X3@X3.T/num_samples
zero_mean = np.zeros(XXT3.shape[0]) 
samples_XXT3 = np.random.multivariate_normal(zero_mean, XXT3, 200).T
XTX3 = X3.T@X3/num_samples

eig_XXT2 = get_eigenspace(XXT2)
eig_XTXT2 = get_eigenspace(XTX2)

eig_XXT3 = get_eigenspace(XXT3)
eig_XTXT3 = get_eigenspace(XTX3)

print("Eigenspace of XXT2: ", eig_XXT2[0],  eig_XXT2[1])
#print("Eigenspace of XTX2: ", eig_XTXT2[0])

print("Eigenspace of XXT23: ", eig_XXT3[0],  eig_XXT3[1])
#print("Eigenspace of XTX3: ", eig_XTXT3[0])

fig_2d, ax_2d =  plt.subplots()
for radius in radii:
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    sampled_i = np.random.choice(range(0, len(x)), 30)
    sampled_x = x[sampled_i]
    sampled_y = y[sampled_i]
    ax_2d.plot(x, y, label=f"Height = {radius}")
ax_2d.scatter(samples_XXT2[0],samples_XXT2[1], color='blue', s=10)
ax_2d.set_aspect('equal')
plt.savefig('../data_images/concentric_circles_2Dcovar_samples.png')





# Create 3D plot of concentric circles with height as radius
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')
for radius in radii:
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(theta, radius)

    
    sampled_i = np.random.choice(range(0, len(x)), 30)
    sampled_x = x[sampled_i]
    sampled_y = y[sampled_i]
    sampled_z = z[sampled_i]

    ax_3d.plot(x, y, z, label=f"Height = {radius}")
#ax_3d.scatter(samples_XXT3[0], samples_XXT3[1], samples_XXT3[2], color='red')

plane_x, plane_y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
plane_z = np.full_like(plane_x, 1.5)  # Plane at height 1.5
ax_3d.plot_surface(plane_x, plane_y, plane_z, color='cyan', alpha=0.5)


ax_3d.set_title("3D Concentric Circles")
ax_3d.legend()
plt.savefig('../data_images/concentric_circles_3D.png')






















