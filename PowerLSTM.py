"""
基于pytorch实现power LSTM的代码，参考文章为：https://arxiv.org/abs/2105.05944
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

'''
power LSTM实现的公式为：
r_t = sigma(U_r X_t + W_r h_{t-1} + b_r)
k_t = r_t @ t + (1-r_t)@k_{t-1}
c^{tilde}_t = tanh(U_c X_t + W_c h_{t-1} + b_c)
f_t = [(t-k_t+1)/(t-k_t+epsilon)]^{-p} #p是 power指数，取值范围在[0, 1], epsilon是小数防止分母为0，取值为0.001
c_t = f_t @ c_{t-1} + i_{t} @ c^{tilde}_{t-1}
i_t = 1 - f_{t}
o_t = sigma(U_o X_t + W_o h_{t-1} + b_o)
h_t = o_t@tanh(c_t)


'''


class pLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, p=None, p_required_learn=False):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.p = p
        if self.p is None and p_required_learn is False:
            self.p = np.random.uniform(0, 1)
        elif p_required_learn is True:
            self.p = nn.Parameter(torch.Tensor(1))
        # r_t = sigma(U_r X_t + W_r h_{t-1} + b_r)
        self.U_r = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_r = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_r = nn.Parameter(torch.Tensor(hidden_sz))
        # c^{tilde}_t = tanh(U_c X_t + W_c h_{t-1} + b_c)
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))
        # o_t = sigma(U_o X_t + W_o h_{t-1} + b_o)
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))
        self.init_weights()  # p的初始化必须在此函数之后，因为init_weights()会初始化p参数，后面再次初始化进行覆盖
        if p_required_learn:
            if torch.cuda.is_available():
                self.p.data = torch.sigmoid(torch.randint(low=-10, high=10, size=(1,), device=torch.device('cuda:0')))
            else:
                self.p.data = torch.sigmoid(torch.randint(low=-10, high=10, size=(1,)))

    def init_weights(self):
        std_v = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.data.dim() >= 2:
                nn.init.xavier_normal_(weight.data)
            elif weight.data.dim != 0:
                nn.init.uniform_(weight.data, -std_v, std_v)

    def forward(self, x, time_sequence, init_states=None, epsilon=0.001):
        """
        设定输入数据张量 x的维度为 [batch_size, sequence_len, feature_size]
        time_seq表示的是输入数据的时间, 维度为[batch_size, sequence_len, 1] 时间是一个维度特征的数据，所以第三个维度为1
        """
        batch_size, seq_len, feature_size = x.shape
        hidden_seq = []
        if init_states is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                h_t, c_t, k_t = (
                    torch.zeros(batch_size, self.hidden_size, device=device, dtype=torch.float64),
                    torch.zeros(batch_size, self.hidden_size, device=device, dtype=torch.float64),
                    torch.zeros(batch_size, self.hidden_size, device=device, dtype=torch.float64)
                )
            else:
                h_t, c_t, k_t = (
                    torch.zeros(batch_size, self.hidden_size, dtype=torch.float64),
                    torch.zeros(batch_size, self.hidden_size, dtype=torch.float64),
                    torch.zeros(batch_size, self.hidden_size, dtype=torch.float64)
                )

        else:
            h_t, c_t, k_t = init_states

        '''
        r_t = sigma(U_r X_t + W_r h_{t-1} + b_r)
        k_t = r_t @ t + (1-r_t)@k_{t-1}
        c^{tilde}_t = tanh(U_c X_t + W_c h_{t-1} + b_c)
        f_t = [(t-k_t+1)/(t-k_t+epsilon)]^{-p} #p是 power指数，取值范围在[0, 1], epsilon是小数防止分母为0，取值为0.001
        c_t = f_t @ c_{t-1} + i_{t} @ c^{tilde}_{t-1}
        i_t = 1 - f_{t}
        o_t = sigma(U_o X_t + W_o h_{t-1} + b_o)
        h_t = o_t@tanh(c_t)
        '''
        for i in range(seq_len):
            '''
            @等价于torch.mm()表示矩阵相乘 * 等价于 torch.mul()表示矩阵对应元素相乘
            '''
            x_t = x[:, i, :]
            t = time_sequence[:, i, :]
            r_t = torch.sigmoid(x_t @ self.U_r + h_t @ self.W_r + self.b_r)
            k_t = r_t * t + (1 - r_t) * k_t
            c_tilde_t = torch.tanh(x_t @ self.U_c + h_t @ self.W_c + self.b_c)
            f_t = torch.pow((t - k_t + 1) / (t - k_t + epsilon), -self.p)
            i_t = 1 - f_t
            c_t = f_t * c_t + i_t * c_tilde_t
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.W_o + self.b_o)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(torch.unsqueeze(h_t, dim=0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()  # contiguous保证了张量在底层存储实在一组连续的内存单元上，增加效率，某些torch的方法要求连续内存
        return hidden_seq, (h_t, c_t, k_t)


class Optimize_pLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, p=None, p_required_learn=False):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.p = p
        if self.p is None and p_required_learn is False:
            self.p = np.random.uniform(0, 1)
        elif p_required_learn is True:
            self.p = nn.Parameter(torch.Tensor(1))
        # r_t = sigma(U_r X_t + W_r h_{t-1} + b_r)
        # self.U_r = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        # self.W_r = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        # self.b_r = nn.Parameter(torch.Tensor(hidden_sz))
        self.U = nn.Parameter(torch.Tensor(input_sz, 3 * hidden_sz))
        self.W = nn.Parameter(torch.Tensor(hidden_sz, 3 * hidden_sz))
        self.b = nn.Parameter(torch.Tensor(3 * hidden_sz))
        self.init_weights()  # p的初始化必须在此函数之后，因为init_weights()会初始化p参数，后面再次初始化进行覆盖
        if p_required_learn:
            if torch.cuda.is_available():
                self.p.data = torch.sigmoid(torch.randint(low=-10, high=10, size=(1,), device=torch.device('cuda:0')))
            else:
                self.p.data = torch.sigmoid(torch.randint(low=-10, high=10, size=(1,)))

    def init_weights(self):
        std_v = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.data.dim() >= 2:
                nn.init.xavier_normal_(weight.data)
            elif weight.data.dim != 0:
                nn.init.uniform_(weight.data, -std_v, std_v)

    def forward(self, x, time_sequence, init_states=None, epsilon=0.001):
        """
        设定输入数据张量 x的维度为 [batch_size, sequence_len, feature_size]
        time_seq表示的是输入数据的时间, 维度为[batch_size, sequence_len, 1] 时间是一个维度特征的数据，所以第三个维度为1
        """
        batch_size, seq_len, feature_size = x.shape
        hidden_seq = []
        if init_states is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                h_t, c_t, k_t = (
                    torch.zeros(batch_size, self.hidden_size, device=device, dtype=torch.float64),
                    torch.zeros(batch_size, self.hidden_size, device=device, dtype=torch.float64),
                    torch.zeros(batch_size, self.hidden_size, device=device, dtype=torch.float64)
                )
            else:
                h_t, c_t, k_t = (
                    torch.zeros(batch_size, self.hidden_size, dtype=torch.float64),
                    torch.zeros(batch_size, self.hidden_size, dtype=torch.float64),
                    torch.zeros(batch_size, self.hidden_size, dtype=torch.float64)
                )

        else:
            h_t, c_t, k_t = init_states

        '''
        r_t = sigma(U_r X_t + W_r h_{t-1} + b_r)
        k_t = r_t @ t + (1-r_t)@k_{t-1}
        c^{tilde}_t = tanh(U_c X_t + W_c h_{t-1} + b_c)
        f_t = [(t-k_t+1)/(t-k_t+epsilon)]^{-p} #p是 power指数，取值范围在[0, 1], epsilon是小数防止分母为0，取值为0.001
        c_t = f_t @ c_{t-1} + i_{t} @ c^{tilde}_{t-1}
        i_t = 1 - f_{t}
        o_t = sigma(U_o X_t + W_o h_{t-1} + b_o)
        h_t = o_t@tanh(c_t)
        '''
        for i in range(seq_len):
            '''
            @等价于torch.mm()表示矩阵相乘 * 等价于 torch.mul()表示矩阵对应元素相乘
            '''
            x_t = x[:, i, :]
            t = time_sequence[:, i, :]
            gates = x_t @ self.U + h_t @ self.W + self.b
            r_t, c_tilde_t, o_t = (
                torch.sigmoid(gates[:, :self.hidden_size]),
                torch.tanh(gates[:, self.hidden_size: 2 * self.hidden_size]),
                torch.sigmoid(gates[:, -self.hidden_size])
            )
            k_t = r_t * t + (1 - r_t) * k_t
            f_t = torch.pow((t - k_t + 1) / (t - k_t + epsilon), -self.p)
            i_t = 1 - f_t
            c_t = f_t * c_t + i_t * c_tilde_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(torch.unsqueeze(h_t, dim=0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()  # contiguous保证了张量在底层存储实在一组连续的内存单元上，增加效率，某些torch的方法要求连续内存
        return hidden_seq, (h_t, c_t, k_t)


# 构建个模型玩一玩
class MyModel(nn.Module):
    def __init__(self, input_features, embedding_dim, output_dim):
        super().__init__()
        self.input_size = input_features
        self.hidden_size = embedding_dim
        self.output_size = output_dim
        self.LSTM = Optimize_pLSTM(self.input_size, self.hidden_size, p=0.25, p_required_learn=False)
        self.Linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, _time, future=0):
        result, (ht, ct, kt) = self.LSTM(x, _time)
        result = self.Linear(result)
        source_pred = result[:, -1, :].reshape(1, 1, self.input_size)
        source_time_pred = _time[:, -1, :].reshape(1, 1, 1)
        delta_t = 0.01
        for _ in range(future):
            result_pred, (ht, ct, kt) = self.LSTM(source_pred, source_time_pred, init_states=(ht, ct, kt))
            result_pred = self.Linear(result_pred)
            source_pred = result_pred
            source_time_pred += delta_t
            result = torch.cat([result, result_pred], dim=1)
        return result




cuda = torch.device('cuda:0')
# power LSTM to learn a sine wave function

data = torch.load('sin_wave.pt', map_location=cuda)
data = data.reshape(1, len(data), 1)
data_train = data[:, :-1, :].reshape(1, 999, 1)
data_target = data[:, 1:, :].reshape(1, 999, 1)
time = torch.load('time_sin_wave.pt', map_location=cuda)
time_seq = time[:-1].reshape(1, len(time) - 1, 1)

model = MyModel(1, 128, 1)
model.to(cuda)
model.double()
model.zero_grad()
criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
loss_record = []
p_record = []
for i in range(500):
    print(f'step:{i}')


    def closure():
        optimizer.zero_grad()
        result = model(data_train, time_seq)
        loss = criterion(data_target, result)
        print(f'training loss==========>{loss.item()}')
        loss_record.append(loss.item())
        loss.backward()
        # name, p_value = list(model.named_parameters())[0]
        # p_record.append(p_value.data.cpu().detach().numpy())
        # print(p_value.data) #当p是可学习的参数时候，可以增加此处的代码，用来检测p的变化
        return loss


    optimizer.step(closure)

torch.save(model.state_dict(), 'pLSTM.params')
plt.figure(1)
plt.plot(loss_record, 'k-', label='training loss', linewidth=2)
plt.show()


def test():
    model_test = MyModel(1, 128, 1)
    model_test.double()
    model_test.to(cuda)
    model_test.load_state_dict(torch.load('pLSTM.params', map_location='cuda'))
    result = model_test(data_train, time_seq, future=100)
    plt.figure()
    plt.plot(data_target.cpu().detach().numpy().reshape(999, ), 'k-', label='real data')
    plt.plot(result.cpu().detach().numpy().reshape(1099, ), 'c-', label='learned data')
    plt.legend(loc='upper right')
    plt.show()


test()
