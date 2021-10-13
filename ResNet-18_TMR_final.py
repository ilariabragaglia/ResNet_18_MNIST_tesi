import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Subset
import xlrd
start_time = time.time()
TRAIN = 0

####################################### DEFINITION OF DEVICE EMPLOY ##########
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
####################################### PARAMETERS ##########
BATCH_SIZE_test = 1
BATCH_SIZE_train = 10
EPOCHS = 3
LEARNING_RATE = 0.001
PATH = './mnist___net.pt'
GRAYSCALE = True
####################################### IMPORT MNIST ##########
trainset = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                                  transform=torchvision.transforms.Compose(
                                                                      [torchvision.transforms.ToTensor(),
                                                                       torchvision.transforms.Normalize((0.1307,),
                                                                                                        (0.3081,))])),
                                       batch_size=BATCH_SIZE_train, shuffle=True)
testset = torchvision.datasets.MNIST('/files/', train=False, download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]))
testset_loader = torch.utils.data.DataLoader(dataset=testset,  batch_size= BATCH_SIZE_test, shuffle=False)


####################################### PREPARE CRITICAL NEURON LIST ##########
flag1 = -1
excel_file = xlrd.open_workbook("Indici_neuroni_ordinamento_merge_Resnet_MNIST2.xlsx", on_demand=True)
sheet0 = excel_file.sheet_by_name("Sheet1")
d = {}

for row in range(150):
    indexes = []
    indexes.append(int(sheet0.cell_value(rowx=row, colx=1)))
    indexes.append(int(sheet0.cell_value(rowx=row, colx=2)))
    indexes.append(int(sheet0.cell_value(rowx=row, colx=3)))
    d[int(sheet0.cell_value(rowx = row, colx = 0))] = indexes

excel_file.release_resources()
del excel_file, sheet0

print(d)

torch.manual_seed(0)
random.seed(0)
np.random.seed((0))


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
        inp = x
        output = self.conv1(inp)
        clock = flag_set()

        if output.shape[1] == 64 and clock == 0:
            for key in d:
                if 15680 < key <= 18816:
                    init = 15680
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority

        elif output.shape[1] == 128 and clock == 0:
            for key in d:
                if 28224 < key <= 30272:
                    init = 28224
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority

        elif output.shape[1] == 256 and clock == 0:
            for key in d:
                if 36416 < key <= 37440:
                    init = 36416
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority


        elif output.shape[1] == 512 and clock == 0:
            for key in d:
                if 40512 < key <= 41024:
                    init = 40512
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority

        elif output.shape[1] == 64 and clock == 2:
            for key in d:
                if 21952 < key <= 25088:
                    init = 21952
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority



        elif output.shape[1] == 128 and clock == 2:
            for key in d:
                if 32320 < key <= 34368:
                    init = 32320
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority


        elif output.shape[1] == 256 and clock == 2:
            for key in d:
                if 38464 < key <= 39488:
                    init = 38464
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority

        elif output.shape[1] == 512 and clock == 2:
            for key in d:
                if 41536 < key <= 42048:
                    init = 41536
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority


        output = self.bn1(output)
        inp = self.relu(output)
        output = self.conv2(inp)
        clock = flag_set()

        if output.shape[1] == 64 and clock == 1:
            for key in d:
                if 18816 < key <= 21952:
                    init = 18816
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority

        elif output.shape[1] == 128 and clock == 1:
            for key in d:
                if 30272 < key <= 32320:
                    init = 30272
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority


        elif output.shape[1] == 256 and clock == 1:
            for key in d:
                if 37440 < key <= 38464:
                    init = 37440
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority


        elif output.shape[1] == 512 and clock == 1:
            for key in d:
                if 41024 < key <= 41536:
                    init = 41024
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority



        elif output.shape[1] == 64 and clock == 3:
            for key in d:
                if 25088 < key <= 28224:
                    init = 25088
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority



        elif output.shape[1] == 128 and clock == 3:
            for key in d:
                if 34368 < key <= 36416:
                    init = 34368
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority



        elif output.shape[1] == 256 and clock == 3:
            for key in d:
                if 39488 < key <= 40512:
                    init = 39488
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority



        elif output.shape[1] == 512 and clock == 3:
            for key in d:
                if 42048 < key <= 42560:
                    init = 42028
                    k1, i, j = d[key][2], d[key][0], d[key][1]
                    Conv = output[0][k1][i][j].item()
                    convolution_comp = Convolution_comp(inp,init,key,clock)
                    Conv1_comp = convolution_comp[0]
                    Conv2_comp =  convolution_comp[1]
                    majority = Voter(Conv,Conv1_comp,Conv2_comp,key)
                    output[0][k1][i][j] = majority

        output = self.bn2(output)
        if self.downsample is not None:
            residual = self.downsample(x)
        output += residual
        output = self.relu(output)

        return output

def flag_set():
    global flag1
    if flag1 < 3:
        flag1 += 1
    else:
        flag1 = 0
    flag = flag1
    return flag

def Weightmatrix_computation(init):
    if init == 0:
        wi, wj, wk = 0, 0, 0
    elif init == 15680:
        wi, wj, wk= 1, 0, 1
    elif init == 28224:
        wi, wj, wk = 2, 0, 1
    elif init == 36416:
        wi, wj, wk = 3, 0, 1
    elif init == 40512:
        wi, wj, wk = 4, 0, 1
    elif init == 21952:
        wi, wj, wk = 1, 1, 1
    elif init == 32320:
        wi, wj, wk = 2, 1, 1
    elif init == 38464:
        wi, wj, wk = 3, 1, 1
    elif init == 41556:
        wi, wj, wk = 4, 1, 1
    elif init == 18816:
        wi, wj, wk = 1, 0, 2
    elif init == 30272:
        wi, wj, wk = 2, 0, 2
    elif init == 37440:
        wi, wj, wk = 3, 0, 2
    elif init == 41024:
        wi, wj, wk = 4, 0, 2
    elif init == 25088:
        wi, wj, wk = 1, 1, 2
    elif init == 34368:
        wi, wj, wk = 2, 1, 2
    elif init == 39488:
        wi, wj, wk = 3, 1, 2
    elif init == 42028:
        wi, wj, wk = 4, 1, 2
    else :
        print('error')
        quit()
    return wi, wj, wk

def Convolution_comp (inp,init,key,clock):
    if clock <= 0 and (init != 15680):
        stride = 2
    else:
        stride = 1
    weightmatrix_computation = Weightmatrix_computation(init)
    wi, wj, wk = weightmatrix_computation[0], weightmatrix_computation[1], weightmatrix_computation[2]
    i, j, k1 = d[key][0], d[key][1], d[key][2]
    channel_ifm = inp.size()[1]
    k_size = inp.size()[2]
    start_sliding = i * k_size * stride + i * stride - i - (stride - 1) * i + j * stride
    inp_unfold = torch.nn.functional.unfold(inp, kernel_size=(k_size, k_size), padding=1)
    tot_unfold_vectors = inp_unfold.size()[1]
    sliding_step = (tot_unfold_vectors / channel_ifm)
    buffer = []
    for pointer in range(channel_ifm):
        buffer.append(start_sliding + sliding_step * pointer)
    set_of_convolved_blocks = tuple(buffer)

    set_convolved_inp_unf = inp_unfold[:, set_of_convolved_blocks, :]
    #print(set_convolved_inp_unf.size())

    set_convolved_inp_unf = torch.reshape(set_convolved_inp_unf, (1, channel_ifm, 3, 3))
    set_convolved_weight_matrices1 = (eval('net.layer{}'.format(int(wi)) + '[{}]'.format(int(wj)) + '.conv{}'.format(
        int(wk)) + '.weight' +'[{},:,:,:]'.format(int(k1))))
    set_convolved_weight_matrices2 = (eval('net.layer{}'.format(int(wi)) + '[{}]'.format(int(wj)) + '.conv{}'.format(
        int(wk)) + '.weight' + '[{},:,:,:]'.format(int(k1))))
    set_convolved_weight_matrices1 = torch.reshape(set_convolved_weight_matrices1, (1, channel_ifm, 3, 3))
    set_convolved_weight_matrices2 = torch.reshape(set_convolved_weight_matrices2, (1, channel_ifm, 3, 3))
    conv1 = torch.nn.functional.conv2d(set_convolved_inp_unf, set_convolved_weight_matrices1)
    conv2 = torch.nn.functional.conv2d(set_convolved_inp_unf, set_convolved_weight_matrices2)
    conv1, conv2 = conv1.item(), conv2.item()
    return conv1,conv2

def Voter(Conv, Conv1_comp, Conv2_comp, key):
    Conv_r = '%.2f' % (Conv)
    Conv1_comp_r = '%.2f' % (Conv1_comp)
    Conv2_comp_r = '%.2f' % (Conv2_comp)

    if Conv_r == Conv1_comp_r == Conv2_comp_r:
        Majority = torch.tensor(Conv)
    elif Conv1_comp_r == Conv2_comp_r and Conv1_comp_r != Conv_r:
        Majority = torch.tensor(Conv1_comp)
        print(key,'no ok', Conv_r, Conv1_comp_r)
    else:
        print('error')
        quit()
    return Majority




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
        inp = x
        output = self.conv1(inp)
        for key in d:
            if 0 < key <= 12544:
                init = 0
                clock = -1
                print('ciao')
                quit()

        x = self.bn1(output)
        x = self.relu(x)
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
    num_classes = 10
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
    num_classes = 10
    net = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, grayscale=GRAYSCALE)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    tot = 10000
    correct = 0
    total = 0

    for i, data in enumerate(testset_loader):
        net.zero_grad()
        start_time1 = time.time()
        inputs, labels = data
        logits, probas= net(inputs)
        _, predicted_labels = torch.max(probas, 1)
        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()
        tot = tot - 1
        print(tot)
        #print("-----%s seconds -----" % (time.time() - start_time1))

    print("accuracy {}%".format(correct / total * 100))
    print("-----%s seconds -----" % (time.time() - start_time))