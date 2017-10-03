# -*- encoding: utf-8 -*-

import torch
import torch.utils.data as Data

# reproducible
torch.manual_seed(1)

# hyper parameters
BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 2
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        # ......
        print('Epoch: ', epoch, '| Step: ', step, '| Batch X: ',
                batch_x.numpy(), '| Batch Y: ', batch_y.numpy())
