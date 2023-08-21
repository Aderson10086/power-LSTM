"""
基于pytorch手搓一个LSTM实现，为后面实现power-LSTM做个练习
LSTM的计算公式为
1.遗忘门 f_t = sigma(U_f X_t + W_f h_{t-1} + b_f)
2.输入门 i_t = sigma(U_i X_t + W_i h_{t-1} + b_i)
3.输出门 o_t = sigma(U_o X_t + W_o h_{t-1} + b_o)
4.更新门 c_t = tanh(U_c X_t + W_c h_{t-1} + b_c)
5.元胞门 c_t = f_t@ c_{t-1} + i_t@c_t
6.状态门 h_t = o_t@tanh(c_t)
"""
import torch
import torch.nn as nn
import numpy as np
import torch.optim as opt


class NaiveCustomLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # 遗忘门
        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))
        # 输入门
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        # 输出门
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))
        # 更新门
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))
        self.init_weights()

    def init_weights(self):
        std_v = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.data.dim() >= 2:
                nn.init.xavier_normal_(weight.data)
            else:
                nn.init.uniform_(weight.data, -std_v, std_v)

    '''
    这里使用了均匀分布的参数初始化方法，可以修改
    nn.init.normal_()
    nn.init.xavier_normal_()
    nn.init.kaiming_uniform_()
    nn.init.kaiming_normal_()
    更多方法可以看 https://pytorch.org/docs/stable/nn.init.html
    '''

    # 前项传播
    def forward(self, x, init_states=None):
        """
        设定输入数据张量 x的维度为 [batch_size, sequence_len, feature_size]
        """
        batch_size, seq_len, feature_size = x.shape
        hidden_seq = []
        if init_states is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                h_t, c_t = (
                    torch.zeros(batch_size, self.hidden_size, device=device),
                    torch.zeros(batch_size, self.hidden_size, device=device)
                )
            else:
                h_t, c_t = (
                    torch.zeros(batch_size, self.hidden_size),
                    torch.zeros(batch_size, self.hidden_size)
                )

        else:
            h_t, c_t = init_states
        for t in range(seq_len):
            x_t = x[:, t, :]  # @表示正常的矩阵相乘和torch.mm等价
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.W_f + self.b_f)
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.W_i + self.b_i)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.W_o + self.b_o)
            c_tilde_t = torch.sigmoid(x_t @ self.U_c + h_t @ self.W_c + self.b_c)
            # * 和 torch.mul()等价表示矩阵对应位置的元素相乘
            c_t = f_t * c_t + i_t * c_tilde_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(torch.unsqueeze(h_t, dim=0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()  # contiguous保证了张量在底层存储实在一组连续的内存单元上，增加效率，某些torch的方法要求连续内存
        return hidden_seq, (h_t, c_t)


# 构建个模型玩玩
class MyModel(nn.Module):
    def __init__(self, n_features, embedding_dim, out_features):
        """

        :param n_features: 输入数据的特征
        :param embedding_dim: 隐藏的维度大小
        :param out_features: 输出的维度大小
        """
        super().__init__()

        self.input_size = n_features
        self.hidden_size = embedding_dim
        self.output_size = out_features
        self.LSTM = NaiveCustomLSTM(input_sz=self.input_size, hidden_sz=self.hidden_size)
        self.Linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x, (_, _) = self.LSTM(x)
        x = self.Linear(x)
        return x


cuda = torch.device('cuda:0')
torch.random.manual_seed(100)
data = torch.randn([2, 2, 2]).to(cuda)
model = MyModel(2, 10, 2)
model.to(cuda)
optimizer = opt.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.MSELoss()
for i in range(300):
    print(f'step:{i}')


    def closure():
        optimizer.zero_grad()
        result = model(data)
        loss = criterion(result, data)
        print(f'loss:{loss}')
        print(f'output:{result}')
        loss.backward()
        return loss


    optimizer.step(closure)

print(data)
