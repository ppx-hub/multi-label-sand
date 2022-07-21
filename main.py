import os
import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import time
import torch.nn as nn
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import cv2
import json
import copy
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='VGG16 for classification in cifar10 Training With Pytorch')
parser.add_argument('--batch_size', default=12, type=int,
                    help='Batch size for training')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                    help='yes or no to choose using warmup strategy to train')
parser.add_argument('--wp_epoch', type=int, default=3,
                    help='The upper bound of warm-up')
parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
args = parser.parse_args()


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.features = nn.ModuleList(base)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [32, 270, 480]
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [64, 135, 240]
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),  # [128, 69, 120]
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),  # [256, 35, 60]
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),  # [256, 18, 30]
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),  # [256, 9, 15]
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*9*15, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 8, bias=False)
        )

        self.last = nn.Sigmoid()
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.features(x)  # 前向传播的时候先经过卷积层和池化层
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)  # 再将features（得到网络输出的特征层）的结果拼接到分类器上
        x = self.last(x)
        return x


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


Label_Map = {
"混乱": 0,
"空洞": 1,
"分裂": 2,
"受伤": 3,
"流动": 4,
"趋中": 5,
"整合": 6,
"能量": 7
}

class MyDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = data.shape[0]

    def __getitem__(self, mask):
        data = self.data[mask]
        label = self.label[mask]
        return data, label

    def __len__(self):
        return self.length


def get_filelist(dir, Imglist, labellist):
    newDir = dir
    tmp_dir = dir
    if os.path.isfile(dir):
        if tmp_dir.split('.')[1] == 'png':
            img = copy.deepcopy(cv2.imdecode(np.fromfile(dir, dtype=np.uint8), 1))
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).float()
            Imglist.append(img)
        elif tmp_dir.split('.')[1] == 'json':
            tmptensor = torch.zeros(8)
            indexlist = []
            with open(dir, 'r', encoding="utf-8") as f:
                data = json.load(f)
                for theme in data["themes"]: # 一个list
                    assert Label_Map[theme["name"]] == theme["type"]
                    indexlist.append(theme["type"])
                tmptensor[indexlist] = 1
                labellist.append(tmptensor)
                tmptensor = 0
        else:
            raise NameError

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Imglist, labellist)

    return Imglist, labellist


if __name__ == '__main__':
    net = ConvNet()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    root = os.path.dirname(__file__)
    train_root = os.path.join(root, "datasets/train")
    test_root = os.path.join(root, "datasets/test")
    X_train, Y_train = get_filelist(train_root, [], [])
    X_train = torch.stack(X_train, dim=0)
    Y_train = torch.stack(Y_train, dim=0)
    train_set = MyDataSet(data=X_train, label=Y_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    X_test, Y_test = get_filelist(test_root, [], [])
    X_test = torch.stack(X_test, dim=0)
    Y_test = torch.stack(Y_test, dim=0)
    test_set = MyDataSet(data=X_test, label=Y_test)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    epoch_size = 50
    base_lr = args.lr
    criterion = nn.BCELoss()  # 定义损失函数：交叉熵
    net.eval() # 训练的时候改为train
    acc = []
    start = time.time()
    losslist = []


    # # 单独的test
    # net.load_state_dict(torch.load('./checkpoint/epoch_25.pth'))
    # correct = 0.0
    # total = 0
    # with torch.no_grad():  # 训练集不需要反向传播
    #     print("=======================test=======================")
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.cuda(), labels.cuda()
    #         pred = (net(inputs) > 0.5).int()
    #
    #         total += inputs.size(0)
    #         correct += int((torch.eq(pred, labels)).sum().item() == labels.shape[1])
    #
    # print("Accuracy of the network on the 19 test images:%.2f %%" % (100 * correct / total))
    # print("===============================================")

    for epoch in range(epoch_size):
        train_loss = 0.0

        # 使用阶梯学习率衰减策略
        if epoch in [30]:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        for iter_i, (inputs, labels) in enumerate(train_loader, 0):
            # 使用warm-up策略来调整早期的学习率
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i + epoch * epoch_size) * 1. / (args.wp_epoch * epoch_size), 4)
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)

            # 将数据从train_loader中读出来,一次读取的样本是32个
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels).cuda()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print('[epoch: %d] loss: %.3f' % (epoch + 1, train_loss / args.batch_size))
        losslist.append(train_loss)
        lr_1 = optimizer.param_groups[0]['lr']
        print("learn_rate:%.15f" % lr_1)
        if epoch % 5 == 4:
            print('Saving epoch %d model ...' % (epoch + 1))
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(net.state_dict(), './checkpoint/epoch_%d.pth' % (epoch + 1))

            # 由于训练集不需要梯度更新,于是进入测试模式
            net.eval()
            correct = 0.0
            total = 0
            with torch.no_grad():  # 训练集不需要反向传播
                print("=======================test=======================")
                for inputs, labels in test_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    pred = (net(inputs) > 0.5).int()

                    total += inputs.size(0)
                    correct += int((torch.eq(pred, labels)).sum().item() == labels.shape[1])

            print("Accuracy of the network on the 19 test images:%.2f %%" % (100 * correct / total))
            print("===============================================")

            acc.append(100 * correct / total)
            net.train()
    print("best acc is %.2f, corresponding epoch is %d" % (max(acc), (np.argmax(acc) + 1) * 5))
    print("===============================================")
    end = time.time()
    print("time:{}".format(end - start))

    ax = np.arange(1, epoch_size + 1)
    plt.plot(ax, losslist)
    plt.xlabel('traing epoch')
    plt.ylabel('traing loss')
    plt.savefig('./loss.png')