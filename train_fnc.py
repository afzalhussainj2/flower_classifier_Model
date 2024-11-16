from torchvision import datasets, transforms
import torch
from torch import nn, optim
from torchvision import models

def loading_data(data_dir):
    """
    Load data from directory
    Args:
    data_dir: str, directory containing data
    Returns:
    trainloader: DataLoader, training data
    """

    # Define transforms
    transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.RandomRotation(30),
                                        transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    data = datasets.ImageFolder(data_dir, transform=transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    
    return dataloaders

def train_model(data_loader, save_dir=None, arch='vgg16', learning_rate=0.001, hidden_units=500, epochs=5, gpu=True):
    """
    Train model
    Args:
    data_loader: DataLoader, training data
    save_dir: str, directory to save model
    arch: str, model architecture
    learning_rate: float, learning rate
    hidden_units: int, number of hidden units
    epochs: int, number of epochs
    gpu: str, use GPU if available
    """
    
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU instead")

    model = models.arch(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),)

    criteria = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in data_loader:
            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            loss = criteria(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in data_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        batch_loss = criteria(output, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(data_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(data_loader):.3f}")
            running_loss = 0
            model.train()
    
    if save_dir:
        # Save the checkpoint
        checkpoint = {'input_size': 25088,
                      'output_size': 102,
                      'classifier': model.classifier,
                      'state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
              'epochs':epochs,
              'learning_rate':optimizer.param_groups[0]['lr']}

        torch.save(checkpoint, 'checkpoint.pth')