# SigmoidCrossEntropy-PyTorch

The example analysis is shown in https://www.jianshu.com/p/ac3bec3dde3e. It equals to BCELoss.

SigmoidCrossEntropy class = BCELoss.

```bash
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.Sigmoid()
)
criterion = nn.BCELoss()

x = torch.randn(16, 10)                 # (16, 10); batch size = 16
y = torch.empty(16, 5).random_(2)       # shape = (16, 5), item = 0 or 1; category number = 5

out = model(x)                          # (16, 5)
out = out.squeeze(dim = -1)             # (16, )

loss = criterion(out, y)
```

```bash
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
criterion = nn.BCEWithLogitsLoss()

x = torch.randn(16, 10)                 # (16, 10)
y = torch.empty(16, 5).random_(2)       # (16, 5)

out = model(x)                          # (16, 5)
out = out.squeeze(dim = -1)             # (16, )

loss = criterion(out, y)
```
