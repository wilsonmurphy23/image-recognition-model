import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from main import Net, create_data_loaders

# Define the predict_image function
def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Define a function to get random images from the test dataset
def get_random_images(num_images, testloader):
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    random_indices = random.sample(range(len(images)), num_images)
    random_images = [images[i] for i in random_indices]
    random_labels = [labels[i] for i in random_indices]
    return random_images, random_labels

# Define a function to display the results
def display_results(num_images, testloader, model):
    to_pil = transforms.ToPILImage()
    random_images, labels = get_random_images(num_images, testloader)
    fig = plt.figure(figsize=(10, 10))
    for ii in range(len(random_images)):
        image = to_pil(random_images[ii])
        index = predict_image(image, model)
        res = int(labels[ii]) == index
        sub = fig.add_subplot(1, len(random_images), ii + 1)
        sub.set_title(f'{testloader.dataset.classes[index]}: {res}')
        plt.axis('off')
        plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    batch_size = 256  # Same batch size as in your main script
    testloader = create_data_loaders(batch_size)[1]  # Get the testloader from your main script
    model = Net()
    model.load_state_dict(torch.load('save_params.ckpt'))
    model.eval()

    # Test the model using random images and display the results
    display_results(8, testloader, model)
