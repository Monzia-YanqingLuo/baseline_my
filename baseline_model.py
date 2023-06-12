##
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy
import matplotlib
import sklearn
import time

##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##
# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the input images to a fixed size
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image tensors
])

# Load the datasets
train_dataset = ImageFolder('/Users/yanqingluo/Desktop/HTCVS2/split_data/train', transform=transform)
test_dataset = ImageFolder('/Users/yanqingluo/Desktop/HTCVS2/split_data/test', transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

##
model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def format_duration(seconds):
    # Calculate the time components
    components = [
        ("w", seconds // 604800),  # 1 week is 604800 seconds
        ("d", seconds // 86400 % 7),  # 1 day is 86400 seconds
        ("h", seconds // 3600 % 24),  # 1 hour is 3600 seconds
        ("min", seconds // 60 % 60),  # 1 minute is 60 seconds
        ("s", round(seconds % 60, 2)),
    ]

    # Only include non-zero components
    components = [(label, value) for label, value in components if value > 0]

    # Format the string
    return ", ".join(f"{value}{label}" for label, value in components)


def print_phase_info(is_training, epoch, total_loss, correct_prediction_count, start_time):
    dataset_length = len(
        training_data_loader.dataset if is_training else testing_data_loader.dataset
    )
    phase_duration = format_duration(time.time() - (training_start_time if is_training else testing_start_time))
    total_duration = format_duration(time.time() - start_time)

    print("    {} Epoch {} done. Loss: {:.2f}, Accuracy: {:.2f}%, Phase Duration: {}, Total Duration: {}".format(
        "Training" if is_training else "Testing",
        epoch,
        total_loss / dataset_length,
        (correct_prediction_count / dataset_length) * 100.0,
        phase_duration,
        total_duration
    ))
start_time = time.time()

for epoch in range(epoch_count):
    epoch_start_time = time.time()
    print("Epoch {} running.".format(epoch))

    # Training Phase
    training_start_time = time.time()
    model.train()
    total_loss = 0.0
    correct_prediction_count = 0

    for inputs, targets in training_data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, targets)

        # Back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct_prediction_count += torch.sum(predictions == targets.data)

    print_phase_info(True, total_loss, correct_prediction_count)

    # Testing Phase
    testing_start_time = time.time()
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        correct_prediction_count = 0

        for inputs, targets in testing_data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            correct_prediction_count += torch.sum(predictions == targets.data)

        print_phase_info(False, total_loss, correct_prediction_count)

    print("Epoch {} done. Epoch Duration: {}, Total Duration: {}".format(
        epoch,
        format_duration(time.time() - epoch_start_time),
        format_duration(time.time() - start_time)
    ))
    print("--------------------------------------------")


##
model.eval()
with torch.no_grad():
    inputs, targets = next(iter(torch.utils.data.DataLoader(testing_data, batch_size=len(testing_data))))
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    confusion_matrix = sklearn.metrics.confusion_matrix(targets, predictions)
    overall_accuracy = confusion_matrix.trace() / confusion_matrix.sum()
    average_accuracy = (confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)).mean()

    confusion_matrix_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    confusion_matrix_display.plot()
    plt.title("Confusion Matrix")
    plt.show()

    print("Overall accuracy: {:.2f}%".format(overall_accuracy * 100))
    print("Average accuracy: {:.2f}%".format(average_accuracy * 100))

