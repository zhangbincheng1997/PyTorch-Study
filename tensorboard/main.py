# https://github.com/lanpa/tensorboardX
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import datasets
from torchvision import transforms
from tensorboardX import SummaryWriter

writer = SummaryWriter('./logs')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset 
dataset = datasets.MNIST(root='./data', 
                                     train=True, 
                                     transform=transforms.ToTensor(),  
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                          batch_size=100, 
                                          shuffle=True)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  

data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
total_step = 10000

# Start training
for step in range(total_step):
    
    # Reset the data_iter
    if (step+1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)

    # Fetch images and labels
    images, labels = next(data_iter)
    inputs, labels = images.view(images.size(0), -1).to(device), labels.to(device)
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute accuracy
    _, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()

    if (step+1) % 100 == 0:
        print ('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}' 
               .format(step+1, total_step, loss.item(), accuracy.item()))

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('accuracy', accuracy.item(), step)

        # 2. Log values and gradients of the parameters (histogram summary)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.cpu().data.numpy(), step)
            writer.add_histogram(name+'/grad', param.grad.cpu().data.numpy(), step)
        
        # 3. Log training images (image summary)
        writer.add_image('images', vutils.make_grid(images.cpu()), step)
