import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from collocation import generate_collocation_points


def plot_data(R, Z, R_colloc, Z_colloc, num_points):
    plt.figure(figsize=(10, 8))
    if not (R[0] == R[-1] and Z[0] == Z[-1]):
        R = np.append(R, R[0])
        Z = np.append(Z, Z[0])
    plt.plot(R, Z, 'b-', linewidth=2, label='Boundary')
    plt.scatter(R_colloc, Z_colloc, s=10, c='r', marker='.', alpha=0.6, label='Collocation Points')
    plt.xlabel(r'$R$', fontsize=14)
    plt.ylabel(r'$Z$', fontsize=14)
    plt.title(f'Helmholtz Domain with {num_points} Collocation Points', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()


a1, a2 = 6, 6
mat_data = io.loadmat(f'boundary_points-a1={a1}-a2={a2}.mat')
R_boundary = mat_data['R_boundary'].flatten()
Z_boundary = mat_data['Z_boundary'].flatten()

num_points = 1000
R_colloc, Z_colloc = generate_collocation_points(R_boundary, Z_boundary, num_points)
io.savemat(f'data_points-a1={a1}-a2={a2}.mat', {
    'R_boundary': R_boundary,
    'Z_boundary': Z_boundary,
    'R_colloc': R_colloc,
    'Z_colloc': Z_colloc
})

plot_data(R_boundary, Z_boundary, R_colloc, Z_colloc, num_points)