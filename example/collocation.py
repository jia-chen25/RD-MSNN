import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.spatial import ConvexHull


def generate_collocation_points(R_boundary, Z_boundary, num_points=1000):
    """
    使用拉丁超立方采样在给定边界内部生成均匀分布的配置点

    参数:
    R_boundary: 边界点的x坐标
    Z_boundary: 边界点的y坐标
    num_points: 要生成的配置点数量

    返回:
    R_colloc: 配置点的x坐标
    Z_colloc: 配置点的y坐标
    """
    # 计算边界的凸包，用于判断点是否在边界内
    points = np.column_stack((R_boundary, Z_boundary))
    hull = ConvexHull(points)
    # 获取边界的范围
    R_min, R_max = np.min(R_boundary), np.max(R_boundary)
    Z_min, Z_max = np.min(Z_boundary), np.max(Z_boundary)
    # 使用拉丁超立方采样生成候选点
    from scipy.stats.qmc import LatinHypercube
    engine = LatinHypercube(d=2)
    candidates = engine.random(n=num_points * 3)  #
    candidates[:, 0] = candidates[:, 0] * (R_max - R_min) + R_min
    candidates[:, 1] = candidates[:, 1] * (Z_max - Z_min) + Z_min
    R_colloc = []
    Z_colloc = []

    for i in range(len(candidates)):
        point = candidates[i]
        # 检查点是否在凸包内
        if is_point_in_convex_hull(point, hull, points):
            R_colloc.append(point[0])
            Z_colloc.append(point[1])
            if len(R_colloc) >= num_points:
                break

    # 如果内部点不足，补充随机点
    while len(R_colloc) < num_points:
        R_rand = np.random.uniform(R_min, R_max)
        Z_rand = np.random.uniform(Z_min, Z_max)
        point = np.array([R_rand, Z_rand])
        if is_point_in_convex_hull(point, hull, points):
            R_colloc.append(R_rand)
            Z_colloc.append(Z_rand)

    return np.array(R_colloc), np.array(Z_colloc)

def is_point_in_convex_hull(point, hull, points):
    """判断点是否在凸包内"""
    new_point = np.append(points, [point], axis=0)
    new_hull = ConvexHull(new_point)
    return len(new_hull.vertices) == len(hull.vertices)


if __name__ == "__main__":
    a1 = 6
    a2 = 6
    mat_data = io.loadmat(f'boundary_points-a1={a1}-a2={a2}.mat')
    R_boundary = mat_data['R_boundary'].flatten()
    Z_boundary = mat_data['Z_boundary'].flatten()
    print(R_boundary)
    print(Z_boundary)
    # 生成配置点
    num_points = 1000
    R_colloc, Z_colloc = generate_collocation_points(R_boundary, Z_boundary, num_points=num_points)
    plt.figure(figsize=(10, 8))
    plt.plot(R_boundary, Z_boundary, 'b-', linewidth=2, label='Boundary')
    plt.scatter(R_colloc, Z_colloc, s=10, c='r', marker='.', alpha=0.6, label='Collocation Points')
    plt.xlabel(r'$\overline{R}$', fontsize=14)
    plt.ylabel(r'$\overline{Z}$', fontsize=14)
    plt.title(f'Magnetic Flux Surface with {num_points} Collocation Points', fontsize=16, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()
    io.savemat(f'collocation_points-a1={a1}-a2={a2}.mat', {'R': R_colloc, 'Z': Z_colloc})
    print(f"已生成 {len(R_colloc)} 个配置点并保存到 collocation_points-a1={a1}-a2={a2}.mat")
