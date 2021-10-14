import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Subset
import xlrd
import time
TRAIN = 0
start_time = time.time()
####################################### DEFINITION OF DEVICE EMPLOY ##########
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
####################################### PARAMETERS ##########
BATCH_SIZE_test = 1
BATCH_SIZE_train = 10  # numero di volte in cui riapplico lo stesso procedimento per immagine
EPOCHS = 3
LEARNING_RATE = 0.001
p = []
input_image = []
classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
classDict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
num_classes = 10
PATH = './mnist___net.pt'
GRAYSCALE = True
torch.manual_seed(0)
random.seed(0)
np.random.seed((0))

####################################### IMPORT MNIST ##########
trainset = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                                  transform=torchvision.transforms.Compose(
                                                                      [torchvision.transforms.ToTensor(),
                                                                       torchvision.transforms.Normalize((0.1307,),
                                                                                                        (0.3081,))])),
                                       batch_size=BATCH_SIZE_train, shuffle=True)
testset = torchvision.datasets.MNIST('/files/', train=False, download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]))
zero_indices, one_indices, two_indices, three_indices, four_indices, five_indices, six_indices, seven_indices, eight_indices, nine_indices = [], [], [], [], [], [], [], [], [], []
testset_loader = torch.utils.data.DataLoader(dataset=testset,  batch_size= BATCH_SIZE_test, shuffle=False)

####################################### PREPARE CRITICAL NEURON LIST ##########
flag1 = -1
excel_file = xlrd.open_workbook("Ordinamento_merge_Resnet_output_neuron_MNIST_corretto.xlsx")
sheet0 = excel_file.sheet_by_name("Sheet1")
critic = []

#for row in range(12000): #1%
 #   critic.append(int(sheet0.cell_value(rowx = row, colx = 0)))
#print(critic)
#excel_file.release_resources()

for cont in range(12000):
    x = random.randint(1,39425)
    critic.append(x)
critic.sort()

#print(critic)



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        clock = flag_set()
        if out.shape[1] == 64 and clock == 0:
            for neuron in critic:
                if 15680 < neuron <= 18816:
                    for i in range(7):
                        for j in range(7):
                            for k in range(64):
                                a = 15680 + (i * 7 * 64 + j * 64 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 128 and clock == 0:
            for neuron in critic:
                if 28224 < neuron <= 30272:
                    for i in range(4):
                        for j in range(4):
                            for k in range(128):
                                a = 28224 + (i * 4 * 128 + j * 128 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 256 and clock == 0:
            for neuron in critic:
                if 36416 < neuron <= 37440:
                    for i in range(2):
                        for j in range(2):
                            for k in range(256):
                                a = 36416 + (i * 2 * 256 + j * 256 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 512 and clock == 0:
            for neuron in critic:
                if 40512 < neuron <= 41024:
                    for i in range(1):
                        for j in range(1):
                            for k in range(512):
                                a = 40512 + (i * 1 * 512 + j * 512 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 64 and clock == 2:
            for neuron in critic:
                if 21952 < neuron <= 25088:
                    for i in range(7):
                        for j in range(7):
                            for k in range(64):
                                a = 21952 + (i * 7 * 64 + j * 64 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0


        if out.shape[1] == 128 and clock == 2:
            for neuron in critic:
                if 32320 < neuron <= 34368:
                    for i in range(4):
                        for j in range(4):
                            for k in range(128):
                                a = 32320 + (i * 4 * 128 + j * 128 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 256 and clock == 2:
            for neuron in critic:
                if 38464 < neuron <= 39488:
                    for i in range(2):
                        for j in range(2):
                            for k in range(256):
                                a = 38464 + (i * 2 * 256 + j * 256 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 512 and clock == 2:
            for neuron in critic:
                if 41536 < neuron <= 42048:
                    for i in range(1):
                        for j in range(1):
                            for k in range(512):
                                a = 41536 + (i * 1 * 512 + j * 512 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        out = self.conv2(out)

        clock = flag_set()
        if out.shape[1] == 64 and clock == 1:
            for neuron in critic:
                if 18816 < neuron <= 21952:
                    for i in range(7):
                        for j in range(7):
                            for k in range(64):
                                c = 18816 + (i * 7 * 64 + j * 64 + k + 1)
                                if neuron == c:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 128 and clock == 1:
            for neuron in critic:
                if 30272 < neuron <= 32320:
                    for i in range(4):
                        for j in range(4):
                            for k in range(128):
                                a = 30272 + (i * 4 * 128 + j * 128 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 256 and clock == 1:
            for neuron in critic:
                if 37440 < neuron <= 38464:
                    for i in range(2):
                        for j in range(2):
                            for k in range(256):
                                a = 37440 + (i * 2 * 256 + j * 256 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 512 and clock == 1:
            for neuron in critic:
                if 41024 < neuron <= 41536:
                    for i in range(1):
                        for j in range(1):
                            for k in range(512):
                                a = 41024 + (i * 1 * 512 + j * 512 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 64 and clock == 3:
            for neuron in critic:
                if 25088 < neuron <= 28224:
                    for i in range(7):
                        for j in range(7):
                            for k in range(64):
                                d = 25088 + (i * 7 * 64 + j * 64 + k + 1)
                                if neuron == d:
                                    out[0][k][i][j] = 0


        if out.shape[1] == 128 and clock == 3:
            for neuron in critic:
                if 34368 < neuron <= 36416:
                    for i in range(4):
                        for j in range(4):
                            for k in range(128):
                                a = 34368 + (i * 4 * 128 + j * 128 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 256 and clock == 3:
            for neuron in critic:
                if 39488 < neuron <= 40512:
                    #print(neuron, 'conv_4_256')
                    for i in range(2):
                        for j in range(2):
                            for k in range(256):
                                a = 39488 + (i * 2 * 256 + j * 256 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        if out.shape[1] == 512 and clock == 3:
            for neuron in critic:
                if 42048 < neuron <= 42560:
                    for i in range(1):
                        for j in range(1):
                            for k in range(512):
                                a = 42048 + (i * 1 * 512 + j * 512 + k + 1)
                                if neuron == a:
                                    out[0][k][i][j] = 0

        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def flag_set():
    global flag1
    if flag1 < 3:
        flag1 += 1
    else:
        flag1 = 0
    flag = flag1
    return flag


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim =3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for neuron in critic:
            if 0 < neuron <= 12544:
                for i in range(14):
                    for j in range(14):
                        for k in range(64):
                            a = (i * 14 * 64 + j * 64 + k + 1)
                            if neuron == a:
                                x[0][k][i][j] = 0

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


if TRAIN == 1:
    net = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, grayscale=GRAYSCALE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    net.train()
    examples = enumerate(testset)
    batch_idx, (example_data, example_labels) = next(examples)
    # print(example_data.shape)
    for epoch in range(EPOCHS):

        for i, data in enumerate(trainset, 0):
            inputs, labels = data
            logits, probas= net(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(net.state_dict(), PATH)
    print('finished Training')

if TRAIN == 0:
    BATCH_SIZE_train = 1

    print("Test mode: \n")
    net = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, grayscale=GRAYSCALE)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    tot = 10000
    correct = 0
    total = 0
    for i, data in enumerate(testset_loader):
        start_time2 = time.time()
        inputs, labels = data
        logits, probas = net(inputs)
        _, predicted_labels = torch.max(probas, 1)
        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()
        #print("-----%s seconds -----" % (time.time() - start_time2))
    print("accuracy {}%".format(correct / total * 100))
    print("-----%s seconds -----"% (time.time()-start_time))
