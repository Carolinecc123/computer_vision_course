from google.colab import drive
drive.mount('/content/gdrive')

import os, sys
path = '/content/gdrive/My Drive/'
dir = os.listdir(path)
for file in dir:
  print(file)

!unzip '/content/gdrive/My Drive/data_semantics.zip'
!unzip '/content/gdrive/My Drive/devkit_semantics.zip'

#!/usr/bin/python
import os, sys
# Open a file
path = "/content/training/image_2"
dirs = sorted(os.listdir(path))
# This would print all the files and directories
length = len(dirs)
train_num = length * 0.7
val_num = length * 0.85
i = 0
train_set = []
val_set = []
test_set = []
#split train, validation and test dataset
for file in dirs:
  i = i + 1
  if i <= train_num:
    train_set.append(file)
  elif i <= val_num:
    val_set.append(file)
  else:
    test_set.append(file) 


#dataset class
import cv2
from torch.utils.data import Dataset
class KittiDataset(Dataset):
  def __init__(self, data_type, transform):
    path = "/content/training/image_2"
    ground_path = '/content/training/semantic'
    self.transform = transform
    self.data = []
  
    if data_type == 'train':
      for file in train_set:
        path = "/content/training/image_2/" + file
        ground_path = '/content/training/semantic/' + file
        self.data.append([path, ground_path])
    #validation dataset
    elif data_type == 'validation':
      for file in val_set:
        path = "/content/training/image_2/" + file
        ground_path = '/content/training/semantic/' + file
        self.data.append([path, ground_path])
    #test dataset
    elif data_type == 'test': 
      for file in val_set:
        path = "/content/training/image_2/" + file
        ground_path = '/content/training/semantic/' + file
        self.data.append([path, ground_path])
    else:
      print('data type input not valid')

    #print(self.data)
      
  def __len__(self):
      return len(self.data)
  def __getitem__(self, idx):
      img_file, ground_truth = self.data[idx]
      #show image path for test dataset

      # read the image file
      #img_pil = Image.open(img_file)    
      image = cv2.imread(img_file)
      truth = cv2.imread(ground_truth)
      if self.transform:
        img_transformed = self.transform(image)
        #truth_transformed = self.transform(truth)
      #print(truth)
      #print(img_file)
      return [img_transformed, truth]


!pip install fcn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pdb

resnet18_pretrained = models.resnet18(pretrained=True)
resnet18_pretrained.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

resnet18_pretrained.fc = nn.Sequential(nn.Conv2d(512, 35, kernel_size=(1, 1), bias=False),
                                       nn.ConvTranspose2d(35, 35, 64, stride=32))
print(resnet18_pretrained)

import os.path as osp
import fcn
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def resnet18(pretrained=True):
   
  #model = torchvision.models.vgg16(pretrained=True)
  if not pretrained:
      return model
  model_file = _get_resnet18_pretrained_model()
  state_dict = torch.load(model_file)
  model.load_state_dict(state_dict)
  return model


def _get_resnet18_pretrained_model():
  model = models.resnet18(pretrained=True)
  model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

  model.fc = nn.Sequential(nn.Conv2d(512, 35, kernel_size=(1, 1), bias=False),
                                        nn.ConvTranspose2d(35, 35, 64, stride=32))
  return model


import os.path as osp
import fcn
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn32s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='https://drive.google.com/uc?id=11k2Q0bvRQgQbT6-jYWeh6nmAsWlSCY3f',  # NOQA
            path=cls.pretrained_model,
            md5='d3eb467a80e7da0468a20dfcbc13e6c8',
        )

    def __init__(self, n_class=21):
        super(FCN32s, self).__init__()
        model = models.resnet18(pretrained=True)
        #resnet 18 layers
        # conv1
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # conv2      
        #zero padding
        #self.pad = nn.ZeroPad2d(100)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(100, 100), bias=False)
        #model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(100, 100), bias=False)
        model_with_multuple_layer = IntermediateLayerGetter(model, {'layer1': 'layer1', 'layer2':'layer2','layer3': 'layer3', 'layer4':'layer4'})
        self.layer = model_with_multuple_layer

        # avgpool
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        # convolution layerss
        self.score_fr = nn.Conv2d(512, 35, kernel_size=(1, 1), bias=False)
        self.upscore = nn.ConvTranspose2d(35, 35, 64, stride=32)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        #h = self.pad(h)

        #intermediate_outputs.append(x)
        h = self.layer(h)
        #print(h)  
        #get output from last layer
        h = h['layer4']
        #print(h)
        
        h = self.avgpool(h)
        h = self.score_fr(h)
        h = self.upscore(h)
        h = h[:, :, 0:x.size()[2], 0:x.size()[3]].contiguous()

        return h

    def copy_params_from_resnet18(self, resnet18):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(resnet18.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = resnet18.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

import os.path as osp

import fcn
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from torchvision.models._utils import IntermediateLayerGetter


#from .fcn32s import get_upsampling_weight

class FCN16s(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn16s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=1bctu58B6YH9bu6lBBSBB2rUeGlGhYLoP',  # NOQA
            path=cls.pretrained_model,
            md5='a2d4035f669f09483b39c9a14a0d6670',
        )

    def __init__(self, n_class=21):
        super(FCN16s, self).__init__()
        model = models.resnet18(pretrained=True)
        # conv1
        #zero padding
        #self.pad = nn.ZeroPad2d(100)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(100, 100), bias=False)
        model_with_multuple_layer = IntermediateLayerGetter(model, {'layer1': 'layer1', 'layer2':'layer2','layer3': 'layer3'})
        self.layer = model_with_multuple_layer
        newmodel = torch.nn.Sequential(*(list(model.children())[-3]))
        self.layer4 = newmodel

        # avgpool
        self.avgpool = nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

        self.score_fr = nn.Conv2d(256, 35, kernel_size=(1, 1), bias=False)
        self.score_fr2 = nn.Conv2d(512, 35, kernel_size=(1, 1), bias=False)
        self.upscore = nn.ConvTranspose2d(35, 35, 4, stride=2)
        self.upscore2 = nn.ConvTranspose2d(35, 35, 32, stride=16)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        #h = self.pad(h)
        h = self.layer(h)

        #get output from previous layer
        h = h['layer3']
        pool4 = h

        #h = self.avgpool(pool4)

        h = self.layer4(h)
        h = self.avgpool(h)
        h = self.score_fr2(h)
        h = self.upscore(h)
        cov5 = h  # 1/16

        pool4 = self.score_fr(pool4)
        cov4 = h[:, :, 0: pool4.size()[2], 0: pool4.size()[3]]

        h = cov4 + cov5
        h = self.upscore2(h)
        h = h[:, :, 29:29+x.size()[2], 29:29+x.size()[3]].contiguous()

        return h

    def copy_params_from_fcn32s(self, fcn32s):
        for name, l1 in fcn32s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)

  

import torchvision.transforms as transforms
import torch
import numpy as np
classes = ('unlabeled','ego vehicle','rectification border','out of roi','static','dynamic','ground','road','sidewalk','parking','rail track','building','wall','fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','background')
transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Resize((375, 1242))])
trainset = KittiDataset(data_type = 'train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)

testset = KittiDataset(data_type = 'test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=2)

print("train2:", len(trainloader.dataset))
#create neural network model
net = FCN32s() 
#net2 = FCN16s()
#print(net2)
#define cross entropy Loss function 
criterion = nn.CrossEntropyLoss()  

#define optimizer    
optimizer = torch.optim.Adam(net2.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

loss_y = []
epoch_x = []
for epoch in range(10):  
  net.train()   
  current_loss = []
  epoch_x.append(epoch + 1)
  running_loss = 0.0

  IoU = 0.0
  true_positive = [0.0] * 35
  false_positive = [0.0] * 35
  false_negative = [0.0] * 35

  for i, data in enumerate(trainloader, 0):
    print(i)
    input_img, truth = data
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(input_img)
    #outputs = net(input_img)
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

  
    #print('imput tensor shape:', input_img.shape)
    print('truth tensor shape:',truth.shape)
    #print(truth)
    truth = truth[:,:,:,0]
    print('truth tensor shape:',truth.shape)
    torch.reshape(truth, (10, 375,1242))
    #torch.reshape(truth, (20, 375,1242))
    #print('truth tensor shape:',truth.shape)
   
    #print('truth tensor shape:',truth.shape)
    truth = truth.long()
    loss = criterion(outputs, truth)
    loss.backward()
    optimizer.step() 

    # print statistics
    running_loss += loss.item()
    current_loss.append(running_loss)
    #print('truth',truth)
    #print('truth shape',truth.shape)
    #print('predict', outputs)
    #print('predict shape',outputs.shape)
    #print('outputs', outputs)
    #pick the max value for predicted class
    predict_max = torch.argmax(outputs, dim=1)
    #print('predict_max.shape', predict_max.shape)
    #print('predict_max',predict_max[0,0,0].item())

    # print('truth.shape', truth.shape)
    # print('truth', truth[0,0,0].item())
    # for i in range(20):
    #   for j in range(375):
    #     for k in range(1242):
    #       predict_value = predict_max[i,j,k].item()
    #       truth_value = truth[i,j,k].item()
    #       if predict_value == truth_value:
    #         true_positive[predict_value] += 1
    #       else:
    #         false_positive[truth_value] += 1

    # print(true_positive)
    # print(false_positive)
    # om = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    # print(om.shape)
    # print(np.unique(om))
    # print('[%d, %5d] loss: %.3f' %
    #               (epoch + 1, i + 1, running_loss))
    running_loss = 0.0
  scheduler.step()  
  #calculate average loss for this epoch
  avg_loss = sum(current_loss)/len(current_loss) 
  loss_y.append(avg_loss)
  print('[%d, %5d] average loss: %.3f' %
                (epoch + 1, i + 1, avg_loss)) 
  

import matplotlib.pyplot as plt
print(epoch_x)
#epoch_x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
print(loss_y)
plt.plot(epoch_x, loss_y)
plt.title("training data loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

import numpy as np

# Let's quickly save our trained model:
PATH = './stl_net.pth'
torch.save(net.state_dict(), PATH)

PATH2 = './stl_net2.pth'
torch.save(net2.state_dict(), PATH2)
# 5. Test the network on the test data
#dataiter = iter(testloader)
#images, truth = dataiter.next()
#print(truth)

net = FCN16s()
net.load_state_dict(torch.load(PATH2))
classes = ('unlabeled','ego vehicle','rectification border','out of roi','static','dynamic','ground','road','sidewalk','parking','rail track','building','wall','fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','background')
#colors = ((  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(111, 74,  0),( 81,  0, 81),(128, 64,128),(244, 35,232),(250,170,160),(230,150,140),( 70, 70, 70),(102,102,156),(190,153,153,(180,165,180),(150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30),(220,220,  0),(107,142, 35),(152,251,152),( 70,130,180),(220, 20, 60),(255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),(  0,  0, 90),(  0,  0,110),(  0, 80,100),(  0,  0,230),(119, 11, 32),(  0,  0,142))
# #model output
# outputs = net(images)
# #print(outputs)

# _, predicted = torch.max(outputs, 1)
#IoU = 0.0
true_positive = [0.0] * 35
false_positive = [0.0] * 35
false_negative = [0.0] * 35
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
  for i, data in enumerate(testloader, 0):
    print(i)
    images, truth = data
    outputs = net(images) 
    truth = truth[:,:,:,0]
    torch.reshape(truth, (10, 375,1242))
    truth = truth.long()
    predict_max = torch.argmax(outputs, dim=1)
    acc, acc_cls, mean_iu, iu = label_accuracy_score(truth.numpy(),predict_max.numpy(),35)
    print('mean_iu',mean_iu)
    print('iu',iu)
  #   predict_max = torch.argmax(outputs, dim=1)
  #   for i in range(10):
  #     for j in range(375):
  #       for k in range(1242):
  #         predict_value = predict_max[i,j,k].item()
  #         truth_value = truth[i,j,k].item()
  #         if predict_value == truth_value:
  #           true_positive[predict_value] += 1
  #         else:
  #           false_positive[predict_value] += 1

  # iou_result = []
  # for num in range(len(predict_value)):
  #   true_pos = true_positive[num]
  #   false_pos = false_positive[num]
  #   #calculate false negative by adding no-current class number
  #   false_neg = 0.0
  #   for tp in range(len(true_positive)):
  #     if tp != num:
  #       false_neg = false_neg + true_positive[tp]
  #   current_iou = true_pos / (true_pos + false_pos + false_neg)  
  #   iou_result.append(current_iou) 

  # #calculate mean iou
  # sum_iou = 0.0
  # for i in range(len(iou_result)):
  #   sum_iou = sum_iou + iou_result[i]
  # mean_iou = sum_iou / len(iou_result)
  # print(mean_iou)
  


# classes = ('unlabeled','ego vehicle','rectification border','out of roi','static','dynamic','ground','road','sidewalk','parking','rail track','building','wall','fence','gruard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','background')
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                                 for j in range(4)))

import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, iu

colors = ((  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(111, 74,  0),( 81,  0, 81),
          (128, 64,128),(244, 35,232),(250,170,160),(230,150,140),( 70, 70, 70),(102,102,156),(190,153,153),
          (180,165,180),(150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30),(220,220,  0),
          (107,142, 35),(152,251,152),( 70,130,180),(220, 20, 60),(255,  0,  0),(  0,  0,142),(  0,  0, 70),
          (  0, 60,100),(  0,  0, 90),(  0,  0,110),(  0, 80,100),(  0,  0,230),(119, 11, 32),(  0,  0,142))


#output image visualization
transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Resize((375, 1242))])
trainset = KittiDataset(data_type = 'train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)

testset = KittiDataset(data_type = 'test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=2)
#colors = ((  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(111, 74,  0),( 81,  0, 81),(128, 64,128),(244, 35,232),(250,170,160),(230,150,140),( 70, 70, 70),(102,102,156),(190,153,153,(180,165,180),(150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30),(220,220,  0),(107,142, 35),(152,251,152),( 70,130,180),(220, 20, 60),(255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),(  0,  0, 90),(  0,  0,110),(  0, 80,100),(  0,  0,230),(119, 11, 32),(  0,  0,142))
with torch.no_grad():
  for i, data in enumerate(testloader, 0):
    images, truth = data
    #print(images)
    outputs = net(images) 
    truth = truth[:,:,:,0]
    torch.reshape(truth, (10, 375,1242))
    truth = truth.long()
    predict_max = torch.argmax(outputs, dim=1)
    #choose target color
    col = 26
    pred = (predict_max == col)
    for image in predict_max:
      img = np.zeros((375,1242,3))
      for i, rows in enumerate(image):
        for j, ind in enumerate(rows):
          if ind > 0:
            #col = predict_max[0][i][j]
            curr_color = colors[col]
            img[i,j,0] = curr_color[0]
            img[i,j,1] = curr_color[1]
            img[i,j,2] = curr_color[2]

      plt.imshow(img)
      #plt.imshow(image)
      plt.show()
