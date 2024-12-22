#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import argparse
#%%

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2,
                    help='Num of layers')
parser.add_argument('--n-test', type=int, default=300,
                    help='Size of test set')
parser.add_argument('--ts-code', type=str, default='601988',
                    help='Stock code')                    
#%%
def create_rolling_window(data, window_size=60):
    """
    创建滚动窗口数据集
    :param data: 原始数据的 DataFrame
    :param window_size: 滚动窗口大小
    :return: 特征矩阵 X 和目标向量 y
    """
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i-window_size:i].drop(columns=['ts_code', 'trade_date', 'pct_chg']).values)
        y.append(data.iloc[i]['pct_chg'])
    X = np.array(X)
    y = np.array(y)
    return X, y
#%%

args, unknown = parser.parse_known_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
# 读取数据
data = pd.read_csv(args.ts_code+'.SH.csv')
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data.sort_values('trade_date', inplace=True)  # 确保数据按日期排序

# 生成滚动窗口数据
window_size = 60
X, y = create_rolling_window(data, window_size)

# 划分训练集和测试集
train_size = len(X) - args.n_test
trainX, testX = X[:train_size], X[train_size:]
trainy, testy = y[:train_size], y[train_size:]

# %%
# # 计算特征数
# feature_num = trainX.shape[2]  # 假设 trainX 的形状为 (样本数, 滚动窗口, 特征数)
# in_dim = window_size * feature_num
out_dim = 1  # 预测一个值，即 pct_change

# 将输入数据从 (样本数, 滚动窗口, 特征数) 展平为 (样本数, 滚动窗口 * 特征数)
# trainX = trainX.reshape(trainX.shape[0], -1)
# testX = testX.reshape(testX.shape[0], -1)

#%%

class Net(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
        self.mamba = Mamba(self.config)
        self.out_proj = nn.Sequential(
            nn.Linear(args.hidden, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    
    def forward(self, x):
        # x 的形状为 (batch_size, seq_len, d_model)
        x = self.mamba(x)  # 输出形状依赖于 Mamba 的实现
        # 取最后一个时间步的输出用于预测
        x = x[:, -1, :]    # 形状为 (batch_size, hidden)
        x = self.out_proj(x)  # 形状为 (batch_size, out_dim)
        return x.squeeze()
    
def PredictWithData(trainX, trainy, testX):
    clf = Net(out_dim=1)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.wd)
    xt = torch.from_numpy(trainX).float()
    xv = torch.from_numpy(testX).float()
    yt = torch.from_numpy(trainy).float()
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    
    for e in range(args.epochs):
        clf.train()
        z = clf(xt)
        loss = F.mse_loss(z, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 10 == 0 and e != 0:
            print('Epoch %d | Loss: %.4f' % (e, loss.item()))
    
    clf.eval()
    mat = clf(xv)
    if args.cuda: mat = mat.cpu()
    yhat = mat.detach().numpy().flatten()
    return yhat
# %%
pred = PredictWithData(trainX, trainy, testX)
# %%
