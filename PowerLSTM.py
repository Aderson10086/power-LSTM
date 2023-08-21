"""
基于pytorch实现power LSTM的代码，参考文章为：https://arxiv.org/abs/2105.05944
"""
import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

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
        self.init_weights()

    def init_weights(self):
        std_v = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.data.dim() >= 2:
                nn.init.xavier_normal_(weight.data)
            else:
                nn.init.uniform_(weight.data, -std_v, std_v)

    def forward(self, x, time_seq, init_states=None, p=None, epsilon=0.001, p_required_learn=False):
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

        if p is None and p_required_learn is False:
            p = np.random.uniform(0, 1)
        elif p_required_learn is True:
            p = nn.Parameter(torch.Tensor(1))
            if torch.cuda.is_available():
                p.data = torch.sigmoid(torch.randint(low=-1000, high=1000, size=(1,), device=torch.device('cuda:0')))
            else:
                p.data = torch.sigmoid(torch.randint(low=-1000, high=1000, size=(1,)))

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
            t = time_seq[:, i, :]
            r_t = torch.sigmoid(x_t @ self.U_r + h_t @ self.W_r + self.b_r)
            k_t = r_t * t + (1 - r_t) * k_t
            c_tilde_t = torch.tanh(x_t @ self.U_c + h_t @ self.W_c + self.b_c)
            f_t = torch.pow((t - k_t + 1) / (t - k_t + epsilon), -p)
            i_t = 1 - f_t
            c_t = f_t * c_t + i_t * c_tilde_t
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.W_o + self.b_o)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(torch.unsqueeze(h_t, dim=0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()  # contiguous保证了张量在底层存储实在一组连续的内存单元上，增加效率，某些torch的方法要求连续内存
        return hidden_seq, (h_t, c_t)


# 构建个模型玩一玩
class MyModel(nn.Module):
    def __init__(self, input_features, embedding_dim, output_dim):
        super().__init__()
        self.input_size = input_features
        self.hidden_size = embedding_dim
        self.output_size = output_dim
        self.LSTM = pLSTM(self.input_size, self.hidden_size)
        self.Linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, time):
        result, (_, _) = self.LSTM(x, time, p_required_learn=False)
        result = self.Linear(result)
        return result


cuda = torch.device('cuda:0')
torch.random.manual_seed(100)
data = torch.randn([2, 2, 2], dtype=torch.float64).to(cuda)
time_seq = torch.tensor([[0.05, 0.1], [0.15, 0.2]], dtype=torch.float64).unsqueeze(dim=2).to(cuda)

model = MyModel(2, 10, 2)
model.double()
model.zero_grad()
model.to(cuda)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
for i in range(300):
    print(f'step:{i}')

    def closure():
        optimizer.zero_grad()
        result = model(data, time_seq)
        loss = criterion(result, time_seq)
        print(f'loss========>{loss}')
        print(f'output:{result}')
        loss.backward()
        return loss
    optimizer.step(closure)

print(data)
'''
loss========>1.9170325807885693e-07
output:tensor([[[ 2.6167e-04, -2.5268e-04],
         [ 1.0006e+00,  9.9939e-01]],

        [[ 9.9968e-01,  1.0003e+00],
         [ 2.0005e+00,  1.9995e+00]]], device='cuda:0', dtype=torch.float64,
       grad_fn=<ViewBackward0>)
'''