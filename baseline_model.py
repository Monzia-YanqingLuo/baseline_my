##
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy
import matplotlib
import sklearn
##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(device)

##
trainData_root = "/Users/yanqingluo/Desktop/HTCVS2/split_data/train"  # 数据集的根文件夹
testData_root = "/Users/yanqingluo/Desktop/HTCVS2/split_data/test"

# 定义预处理转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 标准化
])

# 使用ImageFolder类加载数据集并应用预处理
trainDataset = datasets.ImageFolder(root=trainData_root, transform=transform)
testDataset = datasets.ImageFolder(root=testData_root, transform=transform)
batch_size = 8
shuffle = True

# 创建data loader
trainData_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=shuffle)
testData_loader = DataLoader(testDataset, batch_size=batch_size, shuffle=shuffle)


trainDataset_size = len(trainDataset)
print("Train Dataset size:", trainDataset_size)
testDataset_size = len(testDataset)
print("Test Dataset size:", testDataset_size)
num_train_classes = len(trainDataset.classes)
print("Number of train classes:", num_train_classes)
num_test_classes = len(testDataset.classes)
print("Number of test classes:", num_test_classes)

## model
model = torchvision.models.resnet18(pretrained=False)  #如果要用pretrained，需要先安装ssl
num_classes = 4
model.fc = nn.Linear(model.fc.in_features, num_classes)
## loss function
criterion = nn.CrossEntropyLoss()
##optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


###
num_epochs = 5
learning_rate = 0.001

# Loop over the dataset for the specified number of epochs
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Initialize variables to track loss and accuracy
    total_loss_train = 0.0
    correct_predictions_train = 0

    # Loop over the training dataset in batches
    for inputs, targets in trainData_loader:
        # Move the inputs and targets to the device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update the weights and biases
        optimizer.step()

        # Update the total loss
        total_loss_train += loss.item() * inputs.size(0)

        # Calculate the number of correct predictions
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions_train += (predicted == targets).sum().item()

    # Calculate the average loss and accuracy for the epoch
    average_loss = total_loss_train / trainDataset_size
    accuracy = correct_predictions_train / trainDataset_size * 100.0

    # Print the epoch information
    print("Train Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%".format(epoch+1, num_epochs, average_loss, accuracy))

##
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

model.eval()  # 将模型设置为评估模式

# 初始化变量以跟踪预测结果和真实标签
all_predictions = []
all_targets = []

# 禁用梯度计算
with torch.no_grad():
    for inputs, targets in testData_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 前向传播
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # 保存预测结果和真实标签
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# 生成混淆矩阵
confusion = confusion_matrix(all_targets, all_predictions)

# 显示混淆矩阵
plt.figure(figsize=(10, 8))
plt.imshow(confusion, cmap='Blues')
plt.colorbar()
plt.xticks(np.arange(len(trainDataset.classes)), trainDataset.classes, rotation=90)
plt.yticks(np.arange(len(testDataset.classes)), testDataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
