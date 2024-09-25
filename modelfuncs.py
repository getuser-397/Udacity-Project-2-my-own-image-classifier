import torch
from torch import nn, optim
from torchvision import models
from torchvision.models import VGG16_Weights, ResNet50_Weights

# Function to build the model
def build_model(arch='vgg16', hidden_units=512, output_size=102):
    # Load a pretrained model
    if arch == 'vgg16':
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    elif arch == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Please use a valid architecture: 'vgg16' or 'resnet50'")
    
    # Freeze the parameters to prevent updating
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new classifier
    if arch == 'vgg16':
        input_size = model.classifier[0].in_features
        classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, output_size),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    elif arch == 'resnet50':
        input_size = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, output_size),
            nn.LogSoftmax(dim=1)
        )
    
    return model

# Function to save the checkpoint
 
def save_checkpoint(model, optimizer, path='checkpointpart2.pth', arch='vgg16', hidden_units=512, epochs=1, class_to_idx=None):
    """
    Save the model checkpoint.

    Parameters:
    model (torch.nn.Module): The trained model.
    optimizer (torch.optim.Optimizer): The optimizer used during training.
    path (str): Path where to save the checkpoint.
    arch (str): The architecture of the model.
    hidden_units (int): Number of hidden units in the classifier.
    epochs (int): Number of training epochs.
    class_to_idx (dict): A mapping from class label to index.
    """
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epochs': epochs,
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, path)
    print(f'Model checkpoint saved at {path}')

# Function to load a checkpoint
def load_checkpoint(path='checkpointpart2.pth', learning_rate=0.001, resume_training=False):
    """
    Load a model checkpoint from the specified path and reinitialize the optimizer based on the model's parameters.

    Parameters:
    path (str): Path to the checkpoint file.
    learning_rate (float): Learning rate for reinitializing the optimizer.
    resume_training (bool): If True, will attempt to load the optimizer state for resuming training.

    Returns:
    model (torch.nn.Module): The loaded model.
    optimizer (torch.optim.Optimizer): The optimizer with the loaded state or None if not loaded.
    """
    # Load the saved checkpoint
    checkpoint = torch.load(path)
    
    # Rebuild the model architecture and load the saved state dictionary
    model = build_model(arch=checkpoint['arch'], hidden_units=checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    
    # Set class_to_idx to maintain class-label mapping
    model.class_to_idx = checkpoint['class_to_idx']

    # Reinitialize the optimizer using model's current parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if resume_training:
        try:
            # Attempt to load the optimizer state from the checkpoint
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Optimizer state loaded successfully.")
        except ValueError as e:
            print(f"Error loading optimizer state: {e}. Optimizer state will not be loaded.")
            # The optimizer is reinitialized without loading the state
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Successfully Loaded Model")
    
    # Return both the model and the optimizer (None if not resuming training)
    return model, optimizer if resume_training else None


   

    ...
