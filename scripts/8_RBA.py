import sys
import time
import warnings
from datetime import datetime

from scipy.io import loadmat

from models.model import *
from utils import *

sys.path.insert(0, './utils/')
warnings.filterwarnings('ignore')

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

set_seed(1234)
torch.cuda.empty_cache()


class PhysicsInformedNN():
    """ PINN Class """

    def __init__(self, x, y, t, u, v, p, layers):

        # Data
        X = np.concatenate([x, y, t], axis=1)

        self.lb = torch.tensor(X.min(axis=0)).float().to(device)
        self.ub = torch.tensor(X.max(axis=0)).float().to(device)

        self.X = torch.tensor(X, requires_grad=True).float().to(device)

        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

        self.u = torch.tensor(u).float().to(device)
        self.v = torch.tensor(v).float().to(device)
        self.p = torch.tensor(p).float().to(device)

        self.dnn = DNN(layers).to(device)

        # Optimizer
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(),
            lr=1e-3
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_Adam,
            step_size=5000,
            gamma=0.9
        )

        # History
        self.iter = 0
        self.sum_time = 0.0

        self.loss_history = []
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.loss_log_name = f'../logs/8_RBA_{current_time}'
        self.loss_log_txt = open(self.loss_log_name + '_l2.txt', 'w')
        self.csv_writer = csv.writer(self.loss_log_txt, delimiter=',')

        # Residual-based attention
        self.w_u = self.w_v = 0
        self.eta_u = self.eta_v = 0.01
        self.gamma_u = self.gamma_v = 0.99

    def net_NS(self, x, y, t):
        """ Get the results """

        psi_and_p = self.dnn(torch.cat([x, y, t], dim=1))

        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
        v = - torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

        # Residual calculation
        f_u = u_t + 1.0 * (u * u_x + v * u_y) + p_x - 0.01 * (u_xx + u_yy)
        f_v = v_t + 1.0 * (u * v_x + v * v_y) + p_y - 0.01 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def loss_func(self):
        """ Loss function """

        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(self.x, self.y, self.t)

        mse_s = torch.mean((self.u - u_pred) ** 2) + \
                torch.mean((self.v - v_pred) ** 2)
        mse_r = torch.mean(f_u_pred ** 2) + \
                torch.mean(f_v_pred ** 2)
        mse = mse_s + mse_r

        f_u_norm = self.eta_u * torch.abs(f_u_pred) / torch.max(torch.abs(f_u_pred))
        f_v_norm = self.eta_v * torch.abs(f_v_pred) / torch.max(torch.abs(f_v_pred))

        self.w_u = (self.gamma_u * self.w_u + f_u_norm).detach()
        self.w_v = (self.gamma_v * self.w_v + f_v_norm).detach()

        loss = torch.mean((u_pred - self.u) ** 2) + \
               torch.mean((v_pred - self.v) ** 2) + \
               torch.mean((self.w_u * f_u_pred) ** 2) + \
               torch.mean((self.w_v * f_v_pred) ** 2)

        return loss, mse, mse_s, mse_r

    def train(self, nIter):
        """ Train model """

        self.dnn.train()
        for iter in range(nIter):
            start_time = time.time()

            self.optimizer_Adam.zero_grad()
            loss, mse, mse_s, mse_r = self.loss_func()
            loss.backward(retain_graph=True)
            self.optimizer_Adam.step()
            self.lr_scheduler.step()

            self.iter += 1

            elapsed = time.time() - start_time
            self.sum_time += elapsed

            if iter % 100 == 0:
                print('Iter: %d, Relative l2 error: %.3e, PDE residual: %.3e, Time: %.2f' %
                      (self.iter - 1, mse_s.item(), mse_r.item(), self.sum_time))

                self.loss_history.append(loss.item())
                self.csv_writer.writerow([self.iter - 1, mse_s.item(), mse_r.item(), self.sum_time])

    def predict(self, x_star, y_star, t_star):
        """ Predictions """

        x_star = torch.tensor(x_star, requires_grad=True).float().to(device)
        y_star = torch.tensor(y_star, requires_grad=True).float().to(device)
        t_star = torch.tensor(t_star, requires_grad=True).float().to(device)

        self.dnn.eval()

        u, v, p, _, _ = self.net_NS(x_star, y_star, t_star)

        u_star = u.detach().cpu().numpy()
        v_star = v.detach().cpu().numpy()
        p_star = p.detach().cpu().numpy()

        return u_star, v_star, p_star


if __name__ == "__main__":
    # 设置训练数据点的数量
    N_train = 1000

    # 定义神经网络的结构
    layers = [3, 50, 50, 50, 50, 50, 50, 2]

    # 加载Matlab数据文件中的数据
    data = loadmat('../data/cylinder_nektar_wake.mat')

    # 解析加载的数据
    U_star = data['U_star']  # N x 2 x T (N个数据点，每个点包含两个速度分量的时间序列)
    P_star = data['p_star']  # N x T (N个数据点的压力时间序列)
    t_star = data['t']  # T x 1 (T个时间点)
    X_star = data['X_star']  # N x 2 (N个二维空间坐标点)

    # 获取数据维度信息
    N = X_star.shape[0]  # 数据点的数量
    T = t_star.shape[0]  # 时间点的数量

    # 重新排列数据以适应模型输入要求
    XX = np.tile(X_star[:, 0:1], (1, T))  # 将每个点的x坐标复制T次形成N x T的矩阵
    YY = np.tile(X_star[:, 1:2], (1, T))  # 将每个点的y坐标复制T次形成N x T的矩阵
    TT = np.tile(t_star, (1, N)).T  # 将时间序列复制N次形成N x T的矩阵

    # 提取并展平各个变量以便用于神经网络训练
    UU = U_star[:, 0, :].flatten()[:, None]  # 沿时间轴展平所有点的速度u分量，形成NT x 1的矩阵
    VV = U_star[:, 1, :].flatten()[:, None]  # 沿时间轴展平所有点的速度v分量，形成NT x 1的矩阵
    PP = P_star.flatten()[:, None]  # 沿时间轴展平所有点的压力值，形成NT x 1的矩阵

    # 同样将空间坐标展平
    x = XX.flatten()[:, None]  # NT x 1 的矩阵，包含所有时间步长下的x坐标
    y = YY.flatten()[:, None]  # NT x 1 的矩阵，包含所有时间步长下的y坐标
    t = TT.flatten()[:, None]  # NT x 1 的矩阵，包含所有时间步长下的时间值

    # 最终得到的数据集是 (x, y, t) 对应的空间和时间坐标以及相应的物理量 (u, v, p)
    u = UU.flatten()[:, None]  # NT x 1 的矩阵，包含所有时间步长下的水平速度
    v = VV.flatten()[:, None]  # NT x 1 的矩阵，包含所有时间步长下的竖直速度
    p = PP.flatten()[:, None]  # NT x 1 的矩阵，包含所有时间步长下的压强值

    ######################################################################
    ######################## Noiseless Data ##############################
    ######################################################################

    # 创建训练集：从原始数据集中随机抽取N_u个样本点（不放回抽样）
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train, y_train, t_train = x[idx, :], y[idx, :], t[idx, :]
    u_train, v_train, p_train = u[idx, :], v[idx, :], p[idx, :]

    # 训练模型
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, p_train, layers)
    model.train(1000)

    # 使用Re=100时的精确解作为测试数据
    snap = np.array([100])
    x_test = X_star[:, 0:1]
    y_test = X_star[:, 1:2]
    t_test = TT[:, snap]

    u_test = U_star[:, 0, snap]
    v_test = U_star[:, 1, snap]

    with open(model.loss_log_name + '_uv.txt', 'w') as file:
        # 迭代优化过程(逐步选取另外(1 / 2)样本点)
        for i in range(39):
            u_calc, v_calc, p_calc = model.predict(x_test, y_test, t_test)

            error_u = np.linalg.norm(u_calc - u_test, 2) / np.linalg.norm(u_test, 2)
            error_v = np.linalg.norm(v_calc - v_test, 2) / np.linalg.norm(v_test, 2)
            file.write(f'{model.iter}, {error_u}, {error_v}\n')

            x_old, y_old, t_old = model.x, model.y, model.t
            u_old, v_old, p_old = model.u, model.v, model.p
            _, _, _, f_u_pred, f_v_pred = model.net_NS(x_old, y_old, t_old)
            f_pred = torch.abs(f_u_pred) + torch.abs(f_v_pred)

            mask = f_pred > f_pred.mean()
            x_old, y_old, t_old = x_old[mask].unsqueeze(1), y_old[mask].unsqueeze(1), t_old[mask].unsqueeze(1)
            u_old, v_old, p_old = u_old[mask].unsqueeze(1), v_old[mask].unsqueeze(1), p_old[mask].unsqueeze(1)

            new_idx = np.random.choice(N * T, N_train - mask.sum().item(), replace=False)

            x_new = torch.tensor(x[new_idx, :], requires_grad=True).float().to(device)
            y_new = torch.tensor(y[new_idx, :], requires_grad=True).float().to(device)
            t_new = torch.tensor(t[new_idx, :], requires_grad=True).float().to(device)
            u_new = torch.tensor(u[new_idx, :]).float().to(device)
            v_new = torch.tensor(v[new_idx, :]).float().to(device)
            p_new = torch.tensor(p[new_idx, :]).float().to(device)

            model.x = torch.cat((x_old, x_new), dim=0)
            model.y = torch.cat((y_old, y_new), dim=0)
            model.t = torch.cat((t_old, t_new), dim=0)
            model.u = torch.cat((u_old, u_new), dim=0)
            model.v = torch.cat((v_old, v_new), dim=0)
            model.p = torch.cat((p_old, p_new), dim=0)

            model.train(1000)

    model.loss_log_txt.close()
    plot_l2(model.loss_log_name + '_l2.txt', model.loss_log_name + '_l2' + '.png')
    plot_eq(model.loss_log_name + '_l2.txt', model.loss_log_name + '_eq' + '.png')
    # plot_uv(model.loss_log_name + '_uv.txt', model.loss_log_name + '_uv' + '.png')

    torch.save(model.dnn.state_dict(), '../checkpoints/8_RBA.pth')
    print("Model saved.")

    # 进行模型预测
    snap = np.array([100])
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    t_star = TT[:, snap]

    u_star = U_star[:, 0, snap]
    v_star = U_star[:, 1, snap]
    p_star = P_star[:, snap]

    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)

    plot_solution(X_star, u_pred, 1, "rba_u_pred.png")
    plot_solution(X_star, v_pred, 2, "rba_v_pred.png")
    plot_solution(X_star, p_pred, 3, "rba_p_pred.png")

    plot_solution(X_star, u_star - u_pred, 4, "rba_u_error.png")
    plot_solution(X_star, v_star - v_pred, 5, "rba_v_error.png")
    plot_solution(X_star, p_star - p_pred, 6, "rba_p_error.png")

    plt.show()