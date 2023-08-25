"""
实现时域卷积神经网络
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
torch.random.manual_seed(100)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 添加 weight_norm的作用是为了加速神经网络的训练
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=6, dropout=0.0):
        """
        num_channels表示模型的深度，每层对应一个Block,对于一个序列长度为S的输入数据，模型深度取值为[log_{2}(len(S))]+1,
        如果每层的输出特征都为1，则需要深度个1的list链接成一个[1,1,1,1,1,1,1,1,~~~~~]
        num_inputs是输入时间序列的长度
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        '''
                       ||
                ——————————————-
                | TemporaBlock|
                |-------------|
                       ||
                       ||
                ——————————————-
                | TemporaBlock|
                |-------------|
                       ||
                       ||
        '''

    def forward(self, x):
        return self.network(x)


#
# depth = np.log2((512 - 1) * (2 - 1) / (2 * (7 - 1)) + 1)
# print(depth)
model = TemporalConvNet(num_inputs=1, num_channels=[1, 1, 1, 1, 1, 1])
model.double()
model.zero_grad()
data = torch.load('sin_wave.pt')
data_ = data[:512].reshape(1, 1, 512)
predict_length = 60
target_data = data[predict_length:512+predict_length].reshape(1, 1, 512)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.90)
loss_record = []
for i in range(200):
    print(f'step=>>>>>>{i}')


    def closure():
        optimizer.zero_grad()
        output = model(data_)
        # print(output)
        loss = criterion(output, target_data)
        print(f'loss :{loss.item()}')
        loss_record.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, norm_type=2)
        return loss


    optimizer.step(closure)
    scheduler.step()
    if i == 199:
        with torch.no_grad():
            output = model(data_)
            plt.plot(target_data.reshape(512, ), 'k.', linewidth=1.0, label='real data')
            plt.plot(output.reshape(512, ), 'c-', linewidth=1.0, label='predicted data')
            plt.legend()
            plt.show()

plt.figure()
plt.plot(loss_record)
plt.show()
