import matplotlib.pyplot as plt
from scipy import io
import jax.numpy as jnp
import numpy as np

R = np.linspace(-1, 1, 200)
Z = np.linspace(-1, 1, 200)
R_mesh, Z_mesh = np.meshgrid(R, Z)
print(R_mesh, Z_mesh)


def exact_solution(R, Z, params):
    a1 = params['a1']
    a2 = params['a2']
    u = jnp.sin(a1 * jnp.pi * R) * jnp.sin(a2 * jnp.pi * Z)
    return u

a1 = 6
a2 = 6
lam = 1.0
params_eqn = {'a1': a1, 'a2': a2, 'lam': lam}
psi_values = exact_solution(R_mesh, Z_mesh, params_eqn)
print(exact_solution(0.5, -0.125, params_eqn))

plt.figure(figsize=(10, 8))
levels = np.linspace(np.min(psi_values), np.max(psi_values), 30)
contour = plt.contour(R_mesh, Z_mesh, psi_values,
                      levels=levels,
                      cmap='viridis',
                      linewidths=1.5)
plt.xlabel(r'${R}$', fontsize=14)
plt.ylabel(r'${Z}$', fontsize=14)
plt.title(r'Magnetic Flux Surfaces',
          fontsize=14, pad=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().set_aspect('equal')
cbar = plt.colorbar(contour)
cbar.set_label(r'${\psi}$', fontsize=12)
plt.tight_layout()
plt.show()

def extrude_boundary2():
    # 定义边界范围
    R_min, R_max = -1, 1
    Z_min, Z_max = -1, 1
    # 创建网格点
    x = np.linspace(R_min, R_max, 130)
    z = np.linspace(Z_min, Z_max, 130)
    # 提取四个边界
    top_boundary = np.column_stack([
        x,
        np.full_like(x, Z_max)
    ])
    right_boundary = np.column_stack([
        np.full_like(z, R_max),
        z[::-1]
    ])
    bottom_boundary = np.column_stack([
        x[::-1],
        np.full_like(x, Z_min)
    ])
    left_boundary = np.column_stack([
        np.full_like(z, R_min),
        z
    ])

    all_boundary_points = np.vstack([
        top_boundary,
        right_boundary,
        bottom_boundary,
        left_boundary
    ])
    R_coords = all_boundary_points[:, 0]
    Z_coords = all_boundary_points[:, 1]
    mat_data = {
        'R_boundary': R_coords,
        'Z_boundary': Z_coords,
    }
    io.savemat(f'boundary_points-a1={a1}-a2={a2}.mat', mat_data)
    print(f"成功提取边界点坐标并保存到 boundary_points.mat")
    print(f"总边界点数: {len(R_coords)}")
    print(f"左边界点数: {len(left_boundary)}")
    print(f"右边界点数: {len(right_boundary)}")
    print(f"下边界点数: {len(bottom_boundary)}")
    print(f"上边界点数: {len(top_boundary)}")
    plt.figure(figsize=(10, 8))
    plt.scatter(left_boundary[:, 0], left_boundary[:, 1], s=5, label='left  boundary')
    plt.scatter(right_boundary[:, 0], right_boundary[:, 1], s=5, label='right  boundary')
    plt.scatter(bottom_boundary[:, 0], bottom_boundary[:, 1], s=5, label='up boundary')
    plt.scatter(top_boundary[:, 0], top_boundary[:, 1], s=5, label='down boundary')
    plt.xlabel('R')
    plt.ylabel('Z')
    plt.title('boundary points')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    return all_boundary_points

extrude_boundary2()
