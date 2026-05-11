import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
import os
from jax import random, jit, vjp, grad, vmap
import jax.flatten_util as flat_utl
from tensorflow_probability.substrates import jax as tfp
import time
import functools
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt
from collections import Counter


a1 = 2
a2 = 3
lam = 1.0
params_eqn = {'a1': a1, 'a2': a2, 'lam': lam}
MASTER_CURVE_DIR = Path(f"output/Helmholtz2D-a1={a1}-a2={a2}/ALL_CURVES")
ZERO_CROSS_ROOT = Path(f"output/Helmholtz2D-a1={a1}-a2={a2}/zero-cross")


def get_zero_cross_path(stage_name: str, filename: str = "pde_numzeros.txt") -> str:
    stage_dir = ZERO_CROSS_ROOT / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    return str(stage_dir / filename)


class DualWriter:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'a+', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def create_save_path(base_path):
    create_date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    log_dir = os.path.join(base_path, create_date, "log")
    model_dir = os.path.join(base_path, create_date, "model")
    pic_dir = os.path.join(base_path, create_date, "pic")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pic_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    return log_file, model_dir, pic_dir


save_path = f"output/Helmholtz2D-a1={a1}-a2={a2}/"
log_file_path, model_dir, pic_path = create_save_path(save_path)
sys.stdout = DualWriter(log_file_path)
sys.stderr = sys.stdout

rootdir = Path(__file__).parent
jax.config.update('jax_enable_x64', True)


def save_loss_curve_to_master(loss_all, n_stage):
    loss_all = np.asarray(loss_all)
    total_loss = loss_all[:, 0].astype(np.float64)
    MASTER_CURVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MASTER_CURVE_DIR / f"loss_{n_stage}stage.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(total_loss, f)
    print(f"[save_loss_curve] Saved total loss to: {out_path}")


def init_MLP(parent_key, layer_widths):
    params = []
    keys = random.split(parent_key, num=len(layer_widths) - 1)
    for in_dim, out_dim, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = random.split(key)
        xavier_stddev = jnp.sqrt(2 / (in_dim + out_dim))
        params.append(
            [random.truncated_normal(weight_key, -2, 2, shape=(in_dim, out_dim)) * xavier_stddev,
             random.truncated_normal(bias_key, -2, 2, shape=(out_dim,)) * xavier_stddev]
        )
    return params


def neural_net(params, z, limit, scl, act_s=0):
    actv = [jnp.tanh, jnp.sin][act_s]
    H = z
    first, *hidden, last = params
    H = actv(jnp.dot(H, first[0]) * scl + first[1])
    for layer in hidden:
        H = jnp.tanh(jnp.dot(H, layer[0]) + layer[1])
    var = jnp.dot(H, last[0]) + last[1]
    return var


def sol_init_MLP(parent_key, n_hl, n_unit):
    layers = [2] + n_hl * [n_unit] + [1]
    keys = random.split(parent_key, 1)
    params_u = init_MLP(keys[0], layers)
    return dict(net_u=params_u)


def sol_pred_create(limit, scl, act_s=0):
    def f_u(params, z):
        u = neural_net(params['net_u'], z, limit, scl, act_s)
        return u

    return f_u


def mNN_pred_create(f_u, limit, scl, epsil, act_s=0):
    def f_comb(params, z):
        u_now = neural_net(params['net_u'], z, limit, scl, act_s)
        u = f_u(z) + epsil * u_now
        return u

    return f_comb


def ms_error(diff):
    return jnp.mean(jnp.square(diff), axis=0)


def vgmat(z, n_out, idx=None):
    if idx is None:
        idx = range(n_out)
    n_idx = len(idx)
    n_pt = z.shape[0]
    mat_shape = [n_idx, n_pt, n_out]
    mat = jnp.zeros(mat_shape)
    for l, ii in zip(range(n_idx), idx):
        mat = mat.at[l, :, ii].set(1.)
    return mat


def vectgrad(func, z):
    sol, vjp_fn = vjp(func, z)
    mat = vgmat(z, sol.shape[1])
    grad_sol = vmap(vjp_fn, in_axes=0)(mat)[0]
    n_pd = z.shape[1] * sol.shape[1]
    grad_all = grad_sol.transpose(1, 0, 2).reshape(z.shape[0], n_pd)
    return grad_all, sol


@functools.partial(jit, static_argnames=("lossf", "opt"))
def adam_minimizer(lossf, params, data, opt, opt_state):
    grads, loss_info = grad(lossf, has_aux=True)(params, data)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, loss_info, opt_state


def adam_optimizer(lossf, params, dataf, epoch, lr=1e-3):
    opt_Adam = optax.adam(learning_rate=lr)
    opt_state = opt_Adam.init(params)
    loss_all = []
    data = dataf()
    nc = jnp.int32(jnp.round(epoch / 5))
    nc0 = 2500
    for step in range(epoch):
        params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
        if step % 100 == 0:
            print(f"Step: {step} | Loss: {loss_info[0]:.4e} |"
                  f" Loss_d: {loss_info[1]:.4e} | Loss_e: {loss_info[2]:.4e}")
            data = dataf()

        loss_all.append(loss_info[0:])
        if (step + 1) % (2 * nc0) == 0:
            lossend = np.array(loss_all[-2 * nc0:])[:, 0]
            lc1 = lossend[0: nc0]
            lc2 = lossend[nc0:]
            mm12 = jnp.abs(jnp.mean(lc1) - jnp.mean(lc2))
            stdl2 = jnp.std(lc2)
            if mm12 / stdl2 < 0.4:
                lr = lr / 2
                opt_Adam = optax.adam(learning_rate=lr)
            print(f"learning rate for Adam: {lr:.4e} | mean: {mm12:.3e} | std: {stdl2:.3e}", file=sys.stderr)
    lossend = jnp.array(loss_all[-nc:])[:, 0]
    lmin = jnp.min(lossend)
    llast = lossend[-1]
    while llast > lmin:
        params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
        llast = loss_info[0]
        loss_all.append(loss_info[0:])
    return params, loss_all


def lbfgs_function(lossf, init_params, data):
    _, unflat = flat_utl.ravel_pytree(init_params)

    def update(params_1d):
        return unflat(params_1d)

    loss_history = []

    def f(params_1d):
        params = update(params_1d)
        grads, loss_info = grad(lossf, has_aux=True)(params, data)
        grads_1d = flat_utl.ravel_pytree(grads)[0]
        loss_value = loss_info[0]

        def _callback(loss_vals):
            print(f" Loss: {loss_vals[0]:.4e} |"
                  f" Loss_d: {loss_vals[1]:.4e} | Loss_e: {loss_vals[2]:.4e}")
            loss_history.append(loss_vals.copy())

        jax.debug.callback(_callback, loss_info)
        return loss_value, grads_1d

    f.update = update
    f.loss = loss_history
    return f


def lbfgs_optimizer(lossf, params, data, epoch):
    func_lbfgs = lbfgs_function(lossf, params, data)
    init_params_1d = flat_utl.ravel_pytree(params)[0]
    max_nIter = jnp.int32(epoch / 3)
    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func_lbfgs,
        initial_position=init_params_1d,
        tolerance=1e-10,
        max_iterations=max_nIter
    )
    params = func_lbfgs.update(results.position)
    num_iter = results.num_objective_evaluations
    loss_all = func_lbfgs.loss
    print(f"Total iterations: {num_iter}")
    return params, loss_all


def plot_loss(loss_all, save_dir, fname="training_losses.png"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loss_array = np.array(loss_all)
    total_loss = loss_array[:, 0]
    boundary_loss = loss_array[:, 1]
    equation_loss = loss_array[:, 2]
    boundary_error = loss_array[:, 3]
    f_error = loss_array[:, 4]

    steps = np.arange(len(loss_array))

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.semilogy(steps, total_loss, label='Total Loss', linewidth=2, color='tab:blue')
    plt.semilogy(steps, boundary_loss, label='Boundary Loss', linestyle='--', color='tab:orange')
    plt.semilogy(steps, equation_loss, label='Equation Loss', linestyle='-.', color='tab:green')
    plt.title('Loss Components (Log Scale)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-")

    plt.subplot(2, 1, 2)
    plt.semilogy(steps, boundary_error, label='Boundary Error', linewidth=2, color='tab:red')
    plt.semilogy(steps, f_error, label='Equation Residual', linestyle='--', color='tab:purple')
    plt.title('Error Components (Log Scale)')
    plt.xlabel('Training Steps')
    plt.ylabel('Error Value (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-")

    plt.tight_layout()

    out_path = save_dir / fname
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[plot_loss] Saved loss figure to: {out_path}")


def most_frequent_mean(nums):
    if not nums:
        return None
    counter = Counter(nums)
    max_count = max(counter.values())
    modes = [x for x, cnt in counter.items() if cnt == max_count]
    return sum(modes) / len(modes)


def data_func_create_batched(num_points_per_dim, batch_size=32):
    def dataf():
        x = np.linspace(-1, 1, num_points_per_dim)
        y = np.linspace(-1, 1, num_points_per_dim)
        X, Y = np.meshgrid(x, y)
        z_col = jnp.stack([X.ravel(), Y.ravel()], axis=1)
        key = jax.random.PRNGKey(int(time.time() * 1000) % (2 ** 32))
        indices = jnp.arange(z_col.shape[0])
        indices = jax.random.permutation(key, indices, independent=True)
        num_batches = int(np.ceil(z_col.shape[0] / batch_size))
        batches = jnp.array_split(indices, num_batches)
        return dict(z_col=z_col)

    return dataf


def hessian(func, z):
    grad_func = lambda x: vectgrad(func, x)[0]
    hess_all, grad_sol = vectgrad(grad_func, z)
    return hess_all, grad_sol, func(z)


def gov_eqn(f_u, x, params):
    a1 = params['a1']
    a2 = params['a2']
    lam = params['lam']
    u_x, u = vectgrad(f_u, x)
    u_xx, _, _ = hessian(f_u, x)
    d2u_dx2 = u_xx[:, 0:1]
    d2u_dy2 = u_xx[:, 3:4]
    laplacian_u = d2u_dx2 + d2u_dy2
    u_val = u[:, 0:1]
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    u_exact = jnp.sin(a1 * jnp.pi * x_coord) * jnp.sin(a2 * jnp.pi * y_coord)
    laplacian_exact = - (a1 ** 2 + a2 ** 2) * (jnp.pi ** 2) * u_exact
    f_source = laplacian_exact + lam * u_exact
    residual = laplacian_u + lam * u_val - f_source
    return residual


def loss_create(predf_u, cond, C, lw, loss_ref):
    def loss_fun(params, data):
        f_psi = lambda z: predf_u(params, z)
        z_bnd = cond['cond_nm'][0]
        psi_bnd = cond['cond_nm'][1]
        u_bnd_pred = f_psi(z_bnd)
        bnd_err = ms_error(u_bnd_pred - psi_bnd)
        bnd_err = jnp.hstack([bnd_err])
        z_col = data['z_col']
        f = gov_eqn(f_psi, z_col, C)
        eqn_err_f = jnp.sum(ms_error(f))
        eqn_err = jnp.hstack([eqn_err_f])
        bnd_weight = jnp.array([1.])
        loss_bnd = jnp.sum(bnd_err * bnd_weight)
        loss_eqn = jnp.sum(eqn_err)
        loss = (loss_bnd + lw[0] * loss_eqn) / loss_ref
        loss_info = jnp.hstack([jnp.array([loss, loss_bnd, loss_eqn]),
                                bnd_err, eqn_err])
        return loss, loss_info

    return loss_fun


def exact_solution(z, params):
    return jnp.sin(params['a1'] * jnp.pi * z[:, 0]) * jnp.sin(params['a2'] * jnp.pi * z[:, 1])


def save_training_state(params, loss, filename="training_state.pkl"):
    training_state = {
        'params': params,
        'loss': loss
    }
    with open(filename, 'wb') as f:
        pickle.dump(training_state, f)
    print(f"Training state saved to {filename}")


def load_training_state(filename="training_state.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            training_state = pickle.load(f)
        print(f"Training state loaded from {filename}")
        return training_state['params'], training_state['loss']
    else:
        print(f"No saved training state found at {filename}")
        return None, None


seed = 1234
key = random.PRNGKey(seed)
np.random.seed(seed)
keys = random.split(key, 5)
n_hl = 3
n_unit = 128
scl = 1

N_col = 100
R_min, R_max = -1, 1
Z_min, Z_max = -1, 1
lmt = jnp.array([R_min, R_max, Z_min, Z_max])
epoch1 = 2500
epoch2 = 7500
lw = [0.2, 0.1, 0]

trained_params = sol_init_MLP(keys[0], n_hl, n_unit)
mat_data = loadmat(f'data_points-a1={a1}-a2={a2}.mat')
R_boundary = mat_data['R_boundary'].flatten()
Z_boundary = mat_data['Z_boundary'].flatten()
z_bnd = jnp.stack([R_boundary, Z_boundary], axis=1)
psi_bnd = exact_solution(z_bnd, params_eqn)
cond = dict(cond_nm=[z_bnd, psi_bnd])

dataf = data_func_create_batched(N_col)
data = dataf()
pred_u = sol_pred_create(lmt, scl, act_s=0)
NN_loss = loss_create(pred_u, cond, params_eqn, lw, loss_ref=1)
loss0 = NN_loss(trained_params, data)[0]

"""First stage of training"""
lr = 1e-3
start_time = time.time()
print("adam training")
trained_params, loss1 = adam_optimizer(NN_loss, trained_params, dataf, epoch1, lr=lr)
data = dataf()
print("lbfgs training")
trained_params, loss2 = lbfgs_optimizer(NN_loss, trained_params, data, epoch2)
end_time = time.time()
print('Training time: %.2f' % (end_time - start_time))
model_file1 = os.path.join(model_dir, 'trained_params1.pkl')
save_training_state(trained_params, loss1 + loss2, model_file1)
trained_params, loss_nn1 = load_training_state(model_file1)

f_psi1 = lambda z: pred_u(trained_params, z)
num_points = 1000
R_test = np.linspace(R_min, R_max, num_points)
Z_test = np.linspace(Z_min, Z_max, num_points)
R_grid, Z_grid = np.meshgrid(R_test, Z_test)
z_test = jnp.stack([R_grid.flatten(), Z_grid.flatten()], axis=1)
f1_p = gov_eqn(f_psi1, z_test, params_eqn)
print(jnp.mean(f1_p))
print("Max residual:", jnp.max(jnp.abs(f1_p)))
loss_all = loss_nn1

save_numzeros = []
pde_log_file1 = get_zero_cross_path("stage1", "pde_numzeros_stage1.txt")
with open(pde_log_file1, "w", encoding="utf-8") as f:
    f.write("x方向的零点总数记录\n")
    f1_p_grid = np.array(f1_p.reshape(R_grid.shape))
    for i in range(f1_p_grid.shape[0]):
        row_data = f1_p_grid[i, :]
        idxZero = np.where(row_data[:-1] * row_data[1:] < 0)[0]
        num_zeros = len(idxZero)
        save_numzeros.append(num_zeros)
        f.write(f"第{i}行的零点总数为：{num_zeros}\n")

    f.write("\ny方向的零点总数记录\n")
    for j in range(f1_p_grid.shape[1]):
        col_data = f1_p_grid[:, j]
        idxZero = np.where(col_data[:-1] * col_data[1:] < 0)[0]
        num_zeros = len(idxZero)
        save_numzeros.append(num_zeros)
        f.write(f"第{j}列的零点总数为：{num_zeros}\n")
print(f"[Stage-1] zero-cross saved to: {pde_log_file1}")
print("max_numzeros:", most_frequent_mean(save_numzeros))
NumZero = most_frequent_mean(save_numzeros)
scl2 = 3 * NumZero + 1
print(scl2)
epsil = jnp.array(1e-3)

epoch1 = 2500
epoch2 = 7500
lw = [0.02, 0.001, 0]

dataf2 = data_func_create_batched(N_col)
data2 = dataf2()
trained_params2 = sol_init_MLP(keys[1], n_hl, n_unit)
pred_u2 = mNN_pred_create(f_psi1, lmt, scl2, epsil, act_s=1)
NN_loss2 = loss_create(pred_u2, cond, params_eqn, lw, loss_ref=1)
loss0 = NN_loss2(trained_params2, data2)[0]
print('Loss:', loss0)
NN_loss2 = loss_create(pred_u2, cond, params_eqn, lw, loss_ref=loss0)

print("training the neural network2...")
start_time = time.time()
trained_params2, loss1 = adam_optimizer(NN_loss2, trained_params2, dataf2, epoch1, lr=lr)
data2 = dataf2()
trained_params2, loss2 = lbfgs_optimizer(NN_loss2, trained_params2, data2, epoch2)
model_file2 = os.path.join(model_dir, 'trained_params2.pkl')
save_training_state(trained_params2, loss1 + loss2, model_file2)
trained_params2, loss_nn2 = load_training_state(model_file2)
end_time = time.time()
print("training time2:", end_time - start_time)

f_psi2 = lambda z: pred_u2(trained_params2, z)
f2_p = gov_eqn(f_psi2, z_test, params_eqn)
print("Stage-2 residual mean:", jnp.mean(f2_p))
print("Stage-2 Max residual:", jnp.max(jnp.abs(f2_p)))
loss_all = loss_all + loss_nn2

# ===================== Stage-3：基于 Stage-2 再细化 =====================
save_numzeros_2 = []
run_dir = os.path.dirname(model_dir)
pde_log_file2 = get_zero_cross_path("stage2", "pde_numzeros_stage2.txt")
with open(pde_log_file2, "w", encoding="utf-8") as f:
    f.write("Stage-2 residual: x方向的零点总数记录\n")
    f2_p_grid = np.array(f2_p.reshape(R_grid.shape))
    for i in range(f2_p_grid.shape[0]):
        row_data = f2_p_grid[i, :]
        idxZero = np.where(row_data[:-1] * row_data[1:] < 0)[0]
        num_zeros = len(idxZero)
        save_numzeros_2.append(num_zeros)
        f.write(f"第{i}行的零点总数为：{num_zeros}\n")

    f.write("\nStage-2 residual: y方向的零点总数记录\n")
    for j in range(f2_p_grid.shape[1]):
        col_data = f2_p_grid[:, j]
        idxZero = np.where(col_data[:-1] * col_data[1:] < 0)[0]
        num_zeros = len(idxZero)
        save_numzeros_2.append(num_zeros)
        f.write(f"第{j}列的零点总数为：{num_zeros}\n")
print(f"[Stage-2] zero-cross saved to: {pde_log_file2}")
NumZero2 = most_frequent_mean(save_numzeros_2)
print("Stage-2 max_numzeros (mode mean):", NumZero2)
scl3 = 3 * NumZero2 + 1
print("Stage-3 scl3:", scl3)
epsil3 = jnp.array(1e-3)
epoch1_3 = 2500
epoch2_3 = 7500
lw3 = [0.02, 0.001, 0]
dataf3 = data_func_create_batched(N_col)
data3 = dataf3()
trained_params3 = sol_init_MLP(keys[2], n_hl, n_unit)
pred_u3 = mNN_pred_create(f_psi2, lmt, scl3, epsil3, act_s=1)
NN_loss3 = loss_create(pred_u3, cond, params_eqn, lw3, loss_ref=1)
loss0_3 = NN_loss3(trained_params3, data3)[0]
print("Stage-3 initial Loss:", loss0_3)
NN_loss3 = loss_create(pred_u3, cond, params_eqn, lw3, loss_ref=loss0_3)
print("training the neural network3...")
start_time3 = time.time()
trained_params3, loss1_3 = adam_optimizer(NN_loss3, trained_params3, dataf3, epoch1_3, lr=lr)
data3 = dataf3()
trained_params3, loss2_3 = lbfgs_optimizer(NN_loss3, trained_params3, data3, epoch2_3)
end_time3 = time.time()
print("training time3:", end_time3 - start_time3)

model_file3 = os.path.join(model_dir, 'trained_params3.pkl')
save_training_state(trained_params3, loss1_3 + loss2_3, model_file3)
trained_params3, loss_nn3 = load_training_state(model_file3)
f_psi3 = lambda z: pred_u3(trained_params3, z)
f3_p = gov_eqn(f_psi3, z_test, params_eqn)
print("Stage-3 residual mean:", jnp.mean(f3_p))
print("Stage-3 Max residual:", jnp.max(jnp.abs(f3_p)))
loss_all = loss_all + loss_nn3

# ===================== Stage-4：基于 Stage-3 再细化 =====================

save_numzeros_3 = []
pde_log_file3 = get_zero_cross_path("stage3", "pde_numzeros_stage3.txt")
with open(pde_log_file3, "w", encoding="utf-8") as f:
    f.write("Stage-3 residual: x方向的零点总数记录\n")
    f3_p_grid = np.array(f3_p.reshape(R_grid.shape))
    for i in range(f3_p_grid.shape[0]):
        row_data = f3_p_grid[i, :]
        idxZero = np.where(row_data[:-1] * row_data[1:] < 0)[0]
        num_zeros = len(idxZero)
        save_numzeros_3.append(num_zeros)
        f.write(f"第{i}行的零点总数为：{num_zeros}\n")

    f.write("\nStage-3 residual: y方向的零点总数记录\n")
    for j in range(f3_p_grid.shape[1]):
        col_data = f3_p_grid[:, j]
        idxZero = np.where(col_data[:-1] * col_data[1:] < 0)[0]
        num_zeros = len(idxZero)
        save_numzeros_3.append(num_zeros)
        f.write(f"第{j}列的零点总数为：{num_zeros}\n")
print(f"[Stage-3] zero-cross saved to: {pde_log_file3}")
NumZero3 = most_frequent_mean(save_numzeros_3)
print("Stage-3 max_numzeros (mode mean):", NumZero3)
scl4 = 3 * NumZero3 + 1
print("Stage-4 scl4:", scl4)

epsil4 = jnp.array(1e-3)
epoch1_4 = 2500
epoch2_4 = 7500
lw4 = [0.02, 0.001, 0]
dataf4 = data_func_create_batched(N_col)
data4 = dataf4()
trained_params4 = sol_init_MLP(keys[3], n_hl, n_unit)
pred_u4 = mNN_pred_create(f_psi3, lmt, scl4, epsil4, act_s=1)
NN_loss4 = loss_create(pred_u4, cond, params_eqn, lw4, loss_ref=1)
loss0_4 = NN_loss4(trained_params4, data4)[0]
print("Stage-4 initial Loss:", loss0_4)
NN_loss4 = loss_create(pred_u4, cond, params_eqn, lw4, loss_ref=loss0_4)
print("training the neural network4...")
start_time4 = time.time()
trained_params4, loss1_4 = adam_optimizer(NN_loss4, trained_params4, dataf4, epoch1_4, lr=lr)
data4 = dataf4()
trained_params4, loss2_4 = lbfgs_optimizer(NN_loss4, trained_params4, data4, epoch2_4)
end_time4 = time.time()
print("training time4:", end_time4 - start_time4)

model_file4 = os.path.join(model_dir, 'trained_params4.pkl')
save_training_state(trained_params4, loss1_4 + loss2_4, model_file4)
trained_params4, loss_nn4 = load_training_state(model_file4)
f_psi4 = lambda z: pred_u4(trained_params4, z)
f4_p = gov_eqn(f_psi4, z_test, params_eqn)
print("Stage-4 residual mean:", jnp.mean(f4_p))
print("Stage-4 Max residual:", jnp.max(jnp.abs(f4_p)))
loss_all = loss_all + loss_nn4

loss_pic_dir = os.path.join(run_dir, "loss_pic")
plot_loss(loss_all, save_dir=loss_pic_dir, fname="training_losses.png")
save_loss_curve_to_master(loss_all, n_stage=4)
print("Training completed. All outputs saved to:", run_dir)
