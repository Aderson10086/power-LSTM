import torch
import matplotlib.pyplot as plt
t = torch.linspace(0, 10, 1000, dtype=torch.float64)
y = t + torch.sin(t*10)
plt.figure()
plt.plot(t, y)
plt.show()
torch.save(y, open('sin_wave.pt', 'wb'))
torch.save(t, open('time_sin_wave.pt', 'wb'))