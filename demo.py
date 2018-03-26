import torch
import torchvision #The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision.
import torch.nn as nn #Base class for all neural network modules.
import torch.nn.functional as F #Applies a 1D convolution over an input signal composed of several input planes.
from torchvision import datasets 
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable #Computes and returns the sum of gradients of outputs w.r.t. the inputs.


def to_var(x):
    if torch.cuda.is_available():#utilize the gpu of the pc
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Image processing ToTensor() works for the image, whose elements are in range 0 to 255. You can write your custom Transforms to suit your needs.
transform = transforms.Compose([
                transforms.ToTensor(),    
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5))]) #input[channel] = (input[channel] - mean[channel]) / std[channel](Normalize a tensor image with mean and standard deviation.)
# MNIST dataset root (string) – Root directory of dataset where processed/training.pt and processed/test.pt exist.
#train (bool, optional) – If True, creates dataset from training.pt, otherwise from test.pt.
#transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
mnist = datasets.MNIST(root='./data/',
                       train=True,
                       transform=transform,
                       download=True)
# Data loader
#dataset (Dataset) – dataset from which to load the data.
#batch_size (int, optional) – how many samples per batch to load (default: 1).
#shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=100, 
                                          shuffle=True)
# Discriminator
#nn.Sequential allows you to build a neural net by specifying sequentially the building blocks (nn.Module’s) of that net.
#nn.linear Applies a linear transformation to the incoming data: y=Ax+b--linear(in1_features, in2_features, out_features, bias=True)
#LeakyReLU(negative_slope=0.01, inplace=False)--negative_slope – Controls the angle of the negative slope
#nn.sigmoid --Applies the element-wise function Sigmoid(x)=11+exp(−x)
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid())

# Generator 
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 784),
    nn.Tanh())

if torch.cuda.is_available():
    D.cuda()
    G.cuda()

# Binary cross entropy loss and optimizer
#BCEloss()--Creates a criterion that measures the Binary Cross Entropy between the target and the output:
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)#D.parameters-- are teh parameters to optmize #-lr denotes learning rate
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

# Start training
for epoch in range(200):
    for i, (images, _) in enumerate(data_loader):
        # Build mini-batch dataset
        batch_size = images.size(0)
        images = to_var(images.view(batch_size, -1))
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = to_var(torch.ones(batch_size)) #return a scalar of value 1
        fake_labels = to_var(torch.zeros(batch_size)) #Returns a tensor filled with the scalar value 0

        #============= Train the discriminator =============#
        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop + Optimize
        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        #=============== Train the generator ===============#
        # Compute loss with fake images
        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        g_loss = criterion(outputs, real_labels)
        
        # Backprop + Optimize
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 300 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                  'g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f' 
                  %(epoch, 200, i+1, 600, d_loss.data[0], g_loss.data[0],
                    real_score.data.mean(), fake_score.data.mean()))
    
    # Save real images
    if (epoch+1) == 1:
        images = images.view(images.size(0), 1, 28, 28)
        save_image(denorm(images.data), './data/real_images.png')
    
    # Save sampled images
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images.data), './data/fake_images-%d.png' %(epoch+1))

# Save the trained parameters 
torch.save(G.state_dict(), './generator.pkl')
torch.save(D.state_dict(), './discriminator.pkl')