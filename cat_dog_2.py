import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

##########################################################################
#gets data from dataset and splits into training and testing
#applies preprocessing
#splits the data into mini-batches of size 32
def get_data():
    data_dir = 'Image_Classifier/dataset'

    transform = transforms.Compose([
        #transforms.RandomRotation(20),
        transforms.RandomResizedCrop(128),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])


    train_set = datasets.ImageFolder(data_dir + '/training_set', transform=transform)
    test_set = datasets.ImageFolder(data_dir + '/test_set', transform=transform)

    train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=40, shuffle=True)

    return train_loader, test_loader

#########################################################################

#Define our CNN
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(8, 8)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#########################################################################

#Training the network
def train_net(n_epoch, loss_fn, optimizer, model, train_loader, test_loader, device, scheduler):

    losses = []
    eval_losses = []
    running_loss = 0
    eval_running_loss = 0

    #loop through n_epoch times
    for epoch in range(n_epoch):

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data[0].to(device), data[1].to(device)

            # Set model to train mode
            model.train()
            #Make predictions
            pred = model(inputs)
            
            # Compute loss
            loss = loss_fn(pred, labels)
            # Compute gradients
            loss.backward()
            # Update params and zero grads
            optimizer.step()
            optimizer.zero_grad()
            
            # print statistics
            running_loss += loss.item()
            if i % len(train_loader) == len(train_loader)-1:    # print every len(train_loader) mini-batches
                print('[%d, %5d] loss: %.5f' %(epoch + 1, i + 1, running_loss / len(train_loader)))
                losses.append(running_loss/len(train_loader)) 
                running_loss = 0.0
                scheduler.step()

        with torch.no_grad():
            for  i2, data in enumerate(test_loader, 0):

                inputs, labels = data[0].to(device), data[1].to(device)

                model.eval()

                pred = model(inputs)
                eval_loss = loss_fn(pred, labels)

                eval_running_loss += eval_loss.item()
                if i2 % len(test_loader) == len(test_loader)-1:    # print every len(test_loader) mini-batches
                    print('[%d, %5d] loss: %.5f <--EVAL' %(epoch + 1, i2 + 1, eval_running_loss / len(test_loader)))
                    eval_losses.append(eval_running_loss/len(test_loader)) 
                    eval_running_loss = 0.0

    plt.subplot(1,2,1)
    plt.plot(losses, label='Training loss')

    plt.subplot(1,2,2)
    plt.plot(eval_losses, label='Eval Losses')
    plt.show()

    print('Finished Training')
    
    torch.save(model.state_dict(), PATH)

    print('Saved the Model!')

    return losses, eval_losses, model

#############################################################################

def test_imshow(predicted, images, labels, outputs):
    classes = ('cat', 'dog')
    fig, axes = plt.subplots(figsize = (20,8),nrows =4, ncols = 5)
    plt.subplots_adjust(hspace = 0.48)
    
    for i, a in enumerate(axes.ravel()):
        a.imshow(images[i].permute(1, 2, 0))
        a.title.set_text(''.join('%5s' % classes[labels[i]])+ '  estimate: '+''.join('%5s' % classes[predicted[i]]))
        a.axis('off')
        a.text(40, 150, '%d %%' % np.max(nn.functional.softmax(outputs[i]).numpy()*100), fontsize=10)
    plt.show()

#############################################################################

# testing section

classes = ('cat', 'dog')
#train_loader, test_loader = get_data()

train_loader, test_loader = get_data()

print(len(train_loader))
print(len(test_loader))

################################################################################

# Testing transfer learning with resnet18
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_conv = torchvision.models.resnet18(pretrained=True).to(device)

# Freezes the params in the net
for param in model_conv.parameters():
    param.requires_grad = False

num_filters = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_filters, 2).to(device)

# fc = fully connected, I'm only training the fully connected layer
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
# Using a scheduler to decay lr overr epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#################################################################################

lr = 1e-2

#PATH = 'Image_Classifier/cat_dog_net_2.pth'
PATH = 'Image_Classifier/cat_dog_net_2_resnet.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Net().to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

n_epochs = 25

#train_net(n_epochs, loss_fn, optimizer, model, train_loader, test_loader, device)
model_conv.load_state_dict(torch.load(PATH))

#train_net(n_epochs, loss_fn, optimizer_conv, model_conv, train_loader, test_loader, device, exp_lr_scheduler)

#testing the network
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        model_conv.eval()
        outputs = model_conv(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
           
    test_imshow(predicted.cpu(), images.cpu(), labels.cpu(), outputs.data.cpu())
    for i in range(20):
        print(outputs[i].data.cpu().numpy())

print('Accuracy of the network on the %d test images: %d %%' % (len(test_loader)*20,
    100 * correct / total))



