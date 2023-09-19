import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import logging
torch.backends.mps.is_available()


# Configuration
log_file = 'logfile.log'
num_epochs = 8 #increase this to make it loop more/be more accurate
batch_size = 256  # increase this to make it faster
learning_rate = 0.001

# Set up logging
logging.basicConfig(filename=log_file, level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

#checking if device has proper mps setup to utilize gpu instead of cpu. 
#correct output is tensor([1.], device='mps:0

def check_mps():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        logging.info(x)
    else:
        logging.info("MPS device not found.")

def create_data_loaders(batch_size):
    # Define data transformations for training and testing images.
    # These transformations are applied to the dataset images.
    # For training data:
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly crop the image to 32x32 pixels with 4 pixels of padding.
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally.
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize the image.
    ])

    # For testing data:
    transforms_test = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize the image.
    ])

    # Load the training dataset (CIFAR-10) with specified transformations.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)

    # Create a DataLoader for the training dataset.
    # DataLoader allows you to efficiently iterate through the dataset in batches.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    # - batch_size: Number of samples in each batch for training.
    # - shuffle: Randomly shuffle the dataset for better training.
    # - num_workers: Number of CPU processes to use for data loading (parallelism).

    # Load the testing dataset (CIFAR-10) with specified transformations.
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)

    # Create a DataLoader for the testing dataset.
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    # Similar parameters as the training DataLoader.

    # Return both the training and testing DataLoaders.
    return trainloader, testloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Define the layers and architecture of the neural network.
        # This CNN consists of two convolutional layers followed by two fully connected layers.

        # First convolutional layer:
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # - Input: 3 color channels (RGB), Output: 64 feature maps (output channels),
        #   Kernel size: 3x3, Padding: 1 (to maintain spatial dimensions).

        # Second convolutional layer:
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # - Input: 64 feature maps from the previous layer, Output: 128 feature maps,
        #   Kernel size: 3x3, Padding: 1.

        # First fully connected layer:
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        # - Input: 128x8x8 (flattened feature maps), Output: 256 neurons.

        # Second fully connected layer (output layer):
        self.fc2 = nn.Linear(256, 10)
        # - Input: 256 neurons from the previous layer, Output: 10 neurons (for classification).

    def forward(self, x):
        # Define the forward pass of the neural network.
        
        # First convolutional layer with ReLU activation:
        x = F.relu(self.conv1(x))
        # Apply ReLU activation to the output of the first convolution.
        
        # Max-pooling layer:
        x = F.max_pool2d(x, 2)
        # Apply max-pooling to reduce spatial dimensions by a factor of 2.

        # Second convolutional layer with ReLU activation:
        x = F.relu(self.conv2(x))
        # Apply ReLU activation to the output of the second convolution.
        
        # Max-pooling layer:
        x = F.max_pool2d(x, 2)
        # Apply max-pooling again.

        # Flatten the feature maps:
        x = x.view(-1, 128 * 8 * 8)
        # Reshape the data to be suitable for fully connected layers.

        # First fully connected layer with ReLU activation:
        x = F.relu(self.fc1(x))
        # Apply ReLU activation to the output of the first fully connected layer.

        # Second fully connected layer (output layer):
        x = self.fc2(x)
        # Output the raw scores for classification.

        return x


def train_model(model, trainloader, criterion, optimizer, num_epochs):
    # Training loop for the neural network.

    for epoch in range(num_epochs):
        total_right = 0  # Total number of correct predictions in this epoch.
        total = 0  # Total number of samples processed in this epoch.

        # Iterate over the training dataset in batches.
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            # Wrap inputs and labels as PyTorch Variables for automatic differentiation.

            optimizer.zero_grad()  # Clear gradients from previous iterations.

            outputs = model(inputs)  # Forward pass: compute predictions.
            loss = criterion(outputs, labels)  # Compute the loss.

            loss.backward()  # Backpropagation: compute gradients.
            optimizer.step()  # Update model parameters using the computed gradients.

            _, predicted = torch.max(outputs.data, 1)
            # Find the class indices with the highest predicted scores.
            
            total += labels.size(0)
            # Increment the total count by the batch size.

            total_right += (predicted == labels.data).float().sum()
            # Count how many predictions were correct in this batch.

        training_accuracy = total_right / total
        # Calculate the accuracy for this epoch: correct predictions / total samples.

        logging.info("Training Accuracy for epoch {} : {}".format(epoch + 1, training_accuracy))
        # Log the training accuracy for this epoch.

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), 'save_params.ckpt')
            # Save the model's state_dict every 5 epochs.
            # This allows you to save and restore the model's parameters.

    # Training is complete after all epochs.


def test_model(model, testloader):
    # Evaluation function for the trained neural network on a test dataset.

    # Create a new instance of the neural network.
    my_model = Net()
    my_model.load_state_dict(torch.load('save_params.ckpt'))
    # Load the saved model state_dict to initialize this evaluation model.

    my_model.eval()  # Set the model to evaluation mode.
    # In evaluation mode, certain layers like dropout behave differently.

    total_right = 0  # Total number of correct predictions.
    total = 0  # Total number of samples in the test dataset.

    with torch.no_grad():  # Disable gradient computation for evaluation.
        for data in testloader:  # Iterate over batches in the test dataset.
            images, labels = data
            images, labels = Variable(images), Variable(labels)
            # Wrap test inputs and labels as PyTorch Variables.

            outputs = my_model(images)  # Forward pass: compute predictions.
            _, predicted = torch.max(outputs.data, 1)
            # Find the class indices with the highest predicted scores.

            total += labels.size(0)  # Increment the total count by the batch size.

            total_right += (predicted == labels.data).sum()
            # Count how many predictions were correct in this batch.

    test_accuracy = 100 * total_right / total
    # Calculate the test accuracy: correct predictions / total samples.

    logging.info('Test accuracy: %d %%' % test_accuracy)
    # Log the test accuracy as a percentage.

'''''
if __name__ == "__main__":
    # This block of code will execute when the script is run as the main program.

    check_mps()
    # Check if MPS (Metal Performance Shaders) is available and log its status.

    trainloader, testloader = create_data_loaders(batch_size)
    # Create data loaders for training and testing using the specified batch size.

    model = Net()
    # Create an instance of the neural network model defined in the 'Net' class.

    criterion = nn.CrossEntropyLoss()
    # Define the loss criterion for training, which is typically Cross-Entropy Loss.

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Define the optimizer for training, which is typically Adam with a specified learning rate.

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        # If MPS is available, create an MPS device.

        model.to(mps_device)
        # Move the neural network model to the MPS device for training.

    train_model(model, trainloader, criterion, optimizer, num_epochs)
    # Train the neural network using the specified data loaders, loss criterion, optimizer, and number of epochs.

    test_model(model, testloader)
    # Evaluate the trained model on the test dataset and log the test accuracy.

'''
if __name__ == "__main__":
    check_mps()
    trainloader, testloader = create_data_loaders(batch_size)
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, trainloader, criterion, optimizer, num_epochs)
    test_model(model, testloader)