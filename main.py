#from torch.utils.data import Dataset
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.utils.data import Dataset
import os, sys
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#dataset class
class CustomDataset(Dataset):
    def __init__(self, data_root, transform):
        self.imgs_path = data_root
        self.transform = transform
        self.data = []
        with open(data_root, 'r') as image_file:
            for name in image_file.read().splitlines():
                class_name = name.split(" ")[-1]
                path = name.split(" ")[0]
                self.data.append([path, class_name])
  
        #print(self.data)
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_file, label = self.data[idx]
        #show image path for test dataset
        # if (self.imgs_path == './splits/test.txt'):
        #     print(self.data[idx])

        #print(label)
        # read the image file
        img_pil = Image.open(img_file)
        if self.transform:
            img_transformed = self.transform(img_pil)
        #print(img_transformed, label)    
        return [img_transformed, int(label)]

#Define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #first layer with max pooling
        self.conv1 = nn.Conv2d(3, 6, stride = 1, kernel_size=(5,5))
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)
        #batch normal for first layer
        self.conv1_bn = nn.BatchNorm2d(6)
        #second layer with max pooling
        self.conv2 = nn.Conv2d(6, 16, stride = 1, kernel_size=(5,5))
        self.pool2 = nn.MaxPool2d((2, 2), stride=2)
        #batch normal for second layer
        self.conv2_bn = nn.BatchNorm2d(16)
        #fully connected layer of dimension 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #batch normal for first fully connected layer
        self.fc1_bn = nn.BatchNorm1d(120)
        #fully connected layer of dimension 84
        self.fc2 = nn.Linear(120, 84)
        #batch normal for second fully connected layer
        self.fc2_bn = nn.BatchNorm1d(84)
        #fully connected layer of dimension 10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #first convolution layer with max pooling
        x = self.pool1(F.relu(self.conv1(x)))
        #batch normal version
        #x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))

        #second convolution layer with max pooling
        x = self.pool2(F.relu(self.conv2(x)))
        #batch normal version
        #x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #batch normal version
        #x = F.relu(self.fc1_bn(self.fc1(x)))
        #x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    #load data
    transform_1 = transforms.Compose([
                                    transforms.ToTensor()
                                    ])

    batch_size = 128

    trainset_1 = torchvision.datasets.STL10(root='./data', split='train', transform=transform_1, download=True)
    #trainset_1 = CustomDataset(data_root = './splits/train.txt', transform = transform_1)

    trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=len(trainset_1), shuffle=True, num_workers=2)
    print("train1:" , len(trainloader_1.dataset))


    #calculate training dataset mean and variance
    mean = 0.
    std = 0.
    for images, _ in trainloader_1:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(trainloader_1.dataset)
    std /= len(trainloader_1.dataset)
    print(mean, std)

    #normalize data with dataset mean and variance
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    #train dataset
    trainset = CustomDataset(data_root='./splits/train.txt', transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    print("train2:", len(trainloader.dataset))

    #test dataset
    #testset = torchvision.datasets.STL10(root='./data/test', split='test', transform=transform, download=True)
    test_set = CustomDataset(data_root = './splits/test.txt', transform = transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2)

    #valset = torchvision.datasets.STL10(root='./data', split='unlabeled', transform=transform, download=True)
    valid_set = CustomDataset(data_root = './splits/val.txt', transform = transform)
    valloader = torch.utils.data.DataLoader(valid_set, batch_size=128, shuffle=True, num_workers=2)

    print("valid num:", len(valloader.dataset))
    print("test num:", len(testloader.dataset))

    #classes name
    classes = ('airplane', 'bird', 'car', 'cat',
            'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

    #create neural network model
    net = Net() 
    #define cross entropy Loss function 
    criterion = nn.CrossEntropyLoss()  

    #define optimizer    
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    #L2 regularization, weight decay of 0.03 is picked
    #optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
    #decay the learning rate by 50% every 20 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    #train the network
    epoch_x = []
    loss_y = []
    epoch_x2 = []
    valid_loss = []
    # loop over the dataset for no more than 100 times, 40 is picked as epoch number
    for epoch in range(40):  
        net.train()   
        epoch_x.append(epoch + 1)
        current_loss = []
        current_loss2 = []

        running_loss = 0.0
        running_loss2 = 0.0
        print("before enumerate:", len(trainloader.dataset))
        for i, data in enumerate(trainloader, 0):
            #print(i)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  
            #scheduler.step()

            # print statistics
            running_loss += loss.item()
            current_loss.append(running_loss)
            # print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

        scheduler.step()
        #calculate average loss for this epoch
        avg_loss = sum(current_loss)/len(current_loss) 
        loss_y.append(avg_loss)   
        print('[%d, %5d] average loss: %.3f' %
                    (epoch + 1, i + 1, avg_loss))    
        #trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

        # for every 5 epoch, check validation loss 
        if epoch % 5 == 4: 
            #evaluation mode
            net.eval()   
            epoch_x2.append(epoch + 1)
            for i2, data2 in enumerate(valloader, 0):
                inputs2, labels2 = data2
                #print("label2:", labels2)
        
                outputs2 = net(inputs2)
                loss2 = criterion(outputs2, labels2)
    
                #optimizer.step()  

                running_loss2 += loss2.item()
                current_loss2.append(running_loss2)
                # print('[%d, %5d] validation loss: %.3f' %
                #     (epoch + 1, i2 + 1, running_loss2))
                running_loss2 = 0.0

            #calculate average validation loss for this epoch
            avg_loss2 = sum(current_loss2)/len(current_loss2)  
            valid_loss.append(avg_loss2)  
            print('[%d, %5d] average validation loss: %.3f' %
                        (epoch + 1, i + 1, avg_loss2))       

    print('Finished Training')  
    # print(epoch_x)
    # print(loss_y)
    # print(epoch_x2)
    # print(valid_loss)

    #plot loss and validation loss
    plt.plot(epoch_x, loss_y)
    plt.title("training data loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    plt.plot(epoch_x2, valid_loss)
    plt.title("validation data loss")
    plt.ylabel('validation loss')
    plt.xlabel('epoch')
    plt.show()

    # Let's quickly save our trained model:

    PATH = './stl_net.pth'
    torch.save(net.state_dict(), PATH)

    # 5. Test the network on the test data
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net()
    net.load_state_dict(torch.load(PATH))

    #model output
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))



    correct0 = 0
    total0 = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total0 += labels.size(0)
            correct0 += (predicted == labels).sum().item()

    # print('Accuracy of the network on the 5000 train images: %d %%' % (
    #     100 * correct0 / total0))



    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        print("enumerate test:", len(testloader.dataset))
        # actual = torch.empty()
        # predicted = torch.empty()
        ind = 0
        for data in testloader:
            images, labels = data
            #print("actual label:", labels)
            if ind==0:
                actual = labels
            else:
                actual = torch.cat((actual, labels), 0)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #print("predicted label:", predicted)

            if ind==0:
                predict = predicted
            else:
                predict = torch.cat((predict, predicted), 0)
            correct += (predicted == labels).sum().item()
            ind = 1
        # print("final actual", actual)    
        # print("final actual length", actual.size())    
        # print("final predict", predict)
        # print("final predict length", predict.size()) 
        confusion = confusion_matrix(actual, predict)
        print("confusion matrix:", confusion)

    print('Accuracy of the network on the 5000 test images: %d %%' % (
        100 * correct / total))

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data    
            print("labels:", labels)
            outputs = net(images)    
            _, predictions = torch.max(outputs, 1)
            print("prediction:", predictions)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, 
                                                    accuracy))    

