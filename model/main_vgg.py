import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        # self.bn1_1 = nn.BatchNorm2d()
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.bn1_2 = nn.BatchNorm2d()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        # self.bn2_1 = nn.BatchNorm2d()
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        # self.bn2_2 = nn.BatchNorm2d()

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        # self.bn3_1 = nn.BatchNorm2d()
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        # self.bn3_2 = nn.BatchNorm2d()
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        # self.bn3_3 = nn.BatchNorm2d()

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        # self.bn4_1 = nn.BatchNorm2d()
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn4_2 = nn.BatchNorm2d()
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn4_3 = nn.BatchNorm2d()

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn5_1 = nn.BatchNorm2d()
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn5_2 = nn.BatchNorm2d()
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn5_3 = nn.BatchNorm2d()

        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool(x)

        x = x.view(-1, 512*1*1)
        x = self.fc1(x)
        return x

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        weight = pred.new_ones(pred.size()) * \
            self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def test_accuracy(net, testloader):
    correct = 0
    total = 0
    for i, data in enumerate(tqdm(testloader, desc='accuracy', position=1, leave=False)):
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train(data_dir='data_dir', save_dir='save', learning_rate=1e-3, weight_decay=0.0, label_smoothing=0.1, epoch=100):
    print("[INFO] Get CIFAR-10 dataloader")
    if not Path(data_dir).exists():
        Path(data_dir).mkdir(parents=True)
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=8)
    print("[INFO] Prepare net, optim, lr_scheduler, loss")
    net = VGG16().cuda()
    criterion = LabelSmoothLoss(smoothing=label_smoothing).cuda()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    print("[INFO] Start training")
    writer = SummaryWriter()
    net.train()
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    for epoch in tqdm(range(1, epoch + 1), desc='Train Epoch', position=0, leave=True):
        loss_average = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(tqdm(trainloader, desc='training', position=1, leave=False)):
            # Learn
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Calculate Loss and train accuracy
            with torch.no_grad():
                loss_average += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # Write some values
        writer.add_scalar('loss', loss_average / i, global_step=epoch)
        writer.add_scalar('train_accuracy', 100 * correct / total, global_step=epoch)
        writer.add_scalar('test_accuracy', test_accuracy(net, testloader), global_step=epoch)
        writer.flush()
        # Save checkpoint
        torch.save({
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f"{save_dir}/VGG16-cifar10-epoch_{epoch}.pth")
        # Step scheduler
        scheduler.step()


def test(data_dir='data_dir', checkpoint_path='save/VGG16-cifar10-epoch_50.pth', index=0, batch=1):
    print(f"[INFO] Load checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    net = VGG16().cuda()
    net.load_state_dict(checkpoint['net'])
    print(f"[INFO] Set activation hook")
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    net.conv1_1.register_forward_hook(get_activation('conv1_1'))
    net.conv1_2.register_forward_hook(get_activation('conv1_2'))
    net.fc1.register_forward_hook(get_activation('fc1'))
    # net.fc2.register_forward_hook(get_activation('fc2'))
    # net.fc3.register_forward_hook(get_activation('fc3'))
    print(f"[INFO] Get CIFAR-10 dataloader")
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                             shuffle=False, num_workers=2)
    print(f"[INFO] Run with index {index}")
    for i, data in enumerate(testloader):
        if i * batch <= index and index < (i + 1) * batch:
            with torch.no_grad():
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
            break
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"[INFO] Predict as {classes[predicted.item()]}({predicted.item()})")
    print(f"[INFO] Save conv1_1 with shape: {activation['conv1_1'].shape}")
    np.savetxt(f"save/conv1_1_index_{index}.txt", activation['conv1_1'].cpu().view(64, -1))
    print(f"[INFO] Save conv1_2 with shape: {activation['conv1_2'].shape}")
    np.savetxt(f"save/conv1_2_index_{index}.txt", activation['conv1_2'].cpu().view(64, -1))
    # print(f"[INFO] Save fc1 with shape: {activation['fc1'].shape}")
    # np.savetxt(f"save.v4/fc1_index_{index}.txt", activation['fc1'].cpu().view(10, -1))
    # print(f"[INFO] Save fc2 with shape: {activation['fc2'].shape}")
    # np.savetxt(f"save.v4/fc2_index_{index}.txt", activation['fc2'].cpu().view(84, -1))
    # print(f"[INFO] Save fc3 with shape: {activation['fc3'].shape}")
    # np.savetxt(f"save.v4/fc2_index_{index}.txt", activation['fc2'].cpu().view(84, -1))
    # np.savetxt(f"save.v4/fc3_index_{index}.txt", activation['fc3'].cpu().view(10, -1))
    # np.savetxt(f"save.v4/fc3_index_{index}.txt", activation['fc3'].cpu().view(10, -1))


def convert(checkpoint_path='save/VGG16-cifar10-epoch_50.pth', output_path='save/values_vgg.txt'):

    print(f"[INFO] Load checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    net = VGG16()
    net.load_state_dict(checkpoint['net'])
    print(f"[INFO] Write weights and biases to {output_path}")
    with open(output_path, 'w') as f:
        # Save weights of conv
        f.write(f"conv1_1.weight: {net.conv1_1.weight.shape}\n\n")
        for oc in range(64):
            for ic in range(3):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv1_1.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv1_2.weight: {net.conv1_2.weight.shape}\n\n")
        for oc in range(64):
            for ic in range(64):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv1_2.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv2_1.weight: {net.conv2_1.weight.shape}\n\n")
        for oc in range(128):
            for ic in range(64):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv2_1.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv2_2.weight: {net.conv2_2.weight.shape}\n\n")
        for oc in range(128):
            for ic in range(128):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv2_2.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv3_1.weight: {net.conv3_1.weight.shape}\n\n")
        for oc in range(256):
            for ic in range(128):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv3_1.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv3_2.weight: {net.conv3_2.weight.shape}\n\n")
        for oc in range(256):
            for ic in range(256):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv3_2.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv3_3.weight: {net.conv3_3.weight.shape}\n\n")
        for oc in range(256):
            for ic in range(256):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv3_3.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv4_1.weight: {net.conv4_1.weight.shape}\n\n")
        for oc in range(512):
            for ic in range(256):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv4_1.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv4_2.weight: {net.conv4_2.weight.shape}\n\n")
        for oc in range(512):
            for ic in range(512):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv4_2.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv4_3.weight: {net.conv4_3.weight.shape}\n\n")
        for oc in range(512):
            for ic in range(512):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv4_3.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv5_1.weight: {net.conv5_1.weight.shape}\n\n")
        for oc in range(512):
            for ic in range(512):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv5_1.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv5_2.weight: {net.conv5_2.weight.shape}\n\n")
        for oc in range(512):
            for ic in range(512):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv5_2.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")

        f.write(f"conv5_3.weight: {net.conv5_3.weight.shape}\n\n")
        for oc in range(512):
            for ic in range(512):
                for i in range(3):
                    for j in range(3):
                        f.write(f"{net.conv5_3.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")
        # Save biases of conv
        f.write(f"conv1_1.bias: {net.conv1_1.bias.shape}\n\n")
        for bias in net.conv1_1.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv1_2.bias: {net.conv1_2.bias.shape}\n\n")
        for bias in net.conv1_2.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv2_1.bias: {net.conv2_1.bias.shape}\n\n")
        for bias in net.conv2_1.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv2_2.bias: {net.conv2_2.bias.shape}\n\n")
        for bias in net.conv2_2.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv3_1.bias: {net.conv3_1.bias.shape}\n\n")
        for bias in net.conv3_1.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv3_2.bias: {net.conv3_2.bias.shape}\n\n")
        for bias in net.conv3_2.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv3_3.bias: {net.conv3_3.bias.shape}\n\n")
        for bias in net.conv3_3.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv4_1.bias: {net.conv4_1.bias.shape}\n\n")
        for bias in net.conv4_1.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv4_2.bias: {net.conv4_2.bias.shape}\n\n")
        for bias in net.conv4_2.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv4_3.bias: {net.conv4_3.bias.shape}\n\n")
        for bias in net.conv4_3.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv5_1.bias: {net.conv5_1.bias.shape}\n\n")
        for bias in net.conv5_1.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv5_2.bias: {net.conv5_2.bias.shape}\n\n")
        for bias in net.conv5_2.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        f.write(f"conv5_3.bias: {net.conv5_3.bias.shape}\n\n")
        for bias in net.conv5_3.bias:
            f.write(f"{bias} ")
        f.write("\n\n")

        # Save weights of fc
        f.write(f"fc1.weight: {net.fc1.weight.shape}\n\n")
        for i in range(10):
            for j in range(512):
                f.write(f"{net.fc1.weight[i][j]} ")
            f.write("\n")
        f.write("\n")

        # Save bias of fc
        f.write(f"fc1.bias: {net.fc1.bias.shape}\n\n")
        for i in range(10):
            f.write(f"{net.fc1.bias[i]} ")
        f.write("\n\n")

if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'test': test,
        'convert': convert
    })