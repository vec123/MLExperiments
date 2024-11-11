import numpy as np

def get_eigenspace(X):
    eigenval, eigenvec = np.linalg.eig(X)
    space = [eigenval, eigenvec]
    return space