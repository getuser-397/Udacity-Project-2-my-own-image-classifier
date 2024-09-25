import os
import torch
from torch import nn, optim
import argparse
from imageFuncs import load_data
from modelfuncs import build_model, save_checkpoint

def train_model(data_dir, arch, hidden_units, learning_rate, epochs, gpu, save_dir):
    # Load data
    train_loader, valid_loader, train_dataset = load_data(data_dir)
    
    # Build the model
    model = build_model(arch=arch, hidden_units=hidden_units)
    print('Building model')

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print('Creating optimizer')

    # Set device to GPU if specified
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Using device: {device}')

    # Training loop
    steps = 0
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()

    print('Model training completed successfully')

    # Create the directory to store the checkpoint if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the checkpoint inside the directory
    checkpoint_path = os.path.join(save_dir, 'checkpointpart2.pth')
    save_checkpoint(model, optimizer, path=checkpoint_path, 
                    arch=arch, hidden_units=hidden_units, 
                    epochs=epochs, class_to_idx=train_dataset.class_to_idx)
    print(f'Model checkpoint saved at {checkpoint_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model checkpoint')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (default: vgg16)')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier') 
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')  
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')  
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--save_dir', type=str, default='checkpointpart2', help='Directory to save the model checkpoint')

    args = parser.parse_args()
    
    train_model(args.data_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu, args.save_dir)



