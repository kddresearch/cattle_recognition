'''
Dataloader first; training and testing
define network; frezze layers
define GPU;
define optimizer, loss;
training loss, testing loss;
training accuracy, testing accuracy
'''
import os
import torch
import numpy as np
from torch import nn, optim
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
img_dim = 256
n_classes = 917
iters = 500
log_interval = 1
train_root = "/media/HDD1/Cattle/extreme_clean/train/"
vali_root = "/media/HDD1/Cattle/extreme_clean/valid/"
test_root = "/media/HDD1/Cattle/extreme_clean/test/"
checkpoint_root = "/media/HDD1/Cattle/extreme_clean/checkpoints/"

data_transforms = transforms.Compose([
    transforms.Resize((img_dim,img_dim)),
    transforms.ToTensor(),
])

# dictionary = datasets.ImageFolder(train_root, transform=data_transforms)
# print(dictionary.class_to_idx)

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_root, transform=data_transforms),
    batch_size = 65*3, shuffle=True, **kwargs)

vali_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(vali_root, transform=data_transforms),
    batch_size = 65*3, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(test_root, transform=data_transforms),
    batch_size = 65*3, shuffle=True, **kwargs)



model = models.resnet34(pretrained=True)

for param in model.parameters():
    param.requires_grad = True

# resnet config
# model.fc = nn.Sequential(
#                       nn.Linear(512, 256),
#                       nn.ReLU(),
#                       nn.Dropout(0.4),
#                       nn.Linear(256, n_classes),
#                       nn.LogSoftmax(dim=1))

# model.classifier[6] = nn.Linear(4096,n_classes)
# model.AuxLogits.fc = nn.Linear(768, n_classes)
# model.fc = nn.Linear(1024, n_classes)

model.fc = nn.Linear(512, n_classes)



model = model.to(device)
# criterion = nn.NLLLoss() # resnet
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
all_train_losses = list()
all_test_losses = list()
min_loss = float("inf")
epoch = 0

# load last checkpoint
# last_saved = "/home/kdd/Documents/Cattle/checkpoints/classifier_0.0000.tar"
# if last_saved.endswith(".tar"):
#     checkpoint = torch.load(last_saved, map_location='cpu')
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     all_train_losses = checkpoint['train_losses']
#     all_test_losses = checkpoint['test_losses']
#     min_loss = checkpoint['min_loss']
#     epoch = checkpoint['epoch']


while epoch < iters:

    model.train()
    epoch_train_loss = list()
    train_correct = 0
    train_total = 0

    for batch_idx, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)

        out = model(img)
        loss = criterion(out, label)
        epoch_train_loss.append(loss.item())

        _, predicted = torch.max(out.data, 1)
        train_total += label.size()[0]
        train_correct += (predicted == label).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()))

    mean_train_loss = np.mean(np.array(epoch_train_loss))
    all_train_losses.append(mean_train_loss)
    print('====> Epoch: {} Average training_loss: {:.4f}; Train accuracy: {:.4%}'.format(epoch, mean_train_loss, train_correct / train_total))


    model.eval()
    epoch_test_loss = list()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(vali_loader):
            img = img.to(device)
            label = label.to(device)

            out = model(img)
            loss = criterion(out, label)
            epoch_test_loss.append(loss.item())

            _, predicted = torch.max(out.data, 1)
            test_total += label.size()[0]
            test_correct += (predicted == label).sum().item()

        mean_test_loss = np.mean(np.array(epoch_test_loss))
        all_test_losses.append(mean_test_loss)
        print('====> Epoch: {} Average valid_loss: {:.4f}; Validation accuracy: {:.4%}'.format(epoch, mean_test_loss, test_correct / test_total))


    # save the better model
    if mean_train_loss < min_loss:
        min_loss = mean_train_loss

        for file in os.listdir(checkpoint_root):
            if file.startswith("classifier") and file.endswith(".tar"):
                os.remove(checkpoint_root + file)

        torch.save({
                'epoch':         epoch,
                'state_dict':    model.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'train_losses':  all_train_losses,
                'test_losses':   all_test_losses,
                'min_loss':      min_loss
                }, checkpoint_root + "classifier_" + '{:.4f}'.format(min_loss) + ".tar")

    epoch += 1

# model.eval()
# epoch_test_loss = list()
# test_correct = 0
# test_total = 0
# with torch.no_grad():
#     for batch_idx, (img, label) in enumerate(test_loader):
#         img = img.to(device)
#         label = label.to(device)

#         out = model(img)
#         loss = criterion(out, label)
#         epoch_test_loss.append(loss.item())

#         _, predicted = torch.max(out.data, 1)
#         test_total += label.size()[0]
#         test_correct += (predicted == label).sum().item()

#     mean_test_loss = np.mean(np.array(epoch_test_loss))
#     all_test_losses.append(mean_test_loss)
#     print('====> Epoch: {} Average testing_loss: {:.4f}; testing accuracy: {:.4%}'.format(epoch, mean_test_loss, test_correct / test_total))
