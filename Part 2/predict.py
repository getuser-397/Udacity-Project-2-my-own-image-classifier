import torch
import json
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from modelfuncs import load_checkpoint

# Function to process the image
def process_image(image_path):
    img = Image.open(image_path)
    
    # Resize with aspect ratio maintained
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 255))
    else:
        img.thumbnail((255, 10000))
    
    # Crop to 224x224
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    
    # Normalize the image
    np_image = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose the image for PyTorch
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.tensor(np_image).float()

# Function to predict the class (or classes) of an image
def predict(image_path, model, top_k=5, gpu=False):
    model.eval()
    
    # Process the image
    image = process_image(image_path).unsqueeze_(0)
    
    # Use GPU if available and specified
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)
    
    # Turn off gradients for prediction
    with torch.no_grad():
        output = model.forward(image)
        
    # Get the top K probabilities and classes
    ps = torch.exp(output)
    top_p, top_class = ps.topk(top_k, dim=1)
    
    return top_p.cpu().numpy()[0], top_class.cpu().numpy()[0]

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict flower name from an image and return the top K most likely classes.')
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    
    args = parser.parse_args()
    
    
    # Load the model from checkpoint
    model, _ = load_checkpoint(args.checkpoint) 

    
    # Display the image
    img = Image.open(args.input)
    img.save("displayed_image.jpg")  # Save the image to a file
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()  # Show the image
    
    # Predict the top K classes
    probs, classes = predict(args.input, model, top_k=args.top_k, gpu=args.gpu)
    
    # Map category numbers to names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        
        # Adjust class indices to match the available keys
        labels = [cat_to_name.get(str(cls + 1), "Unknown") for cls in classes]  # Add 1 if necessary
    else:
        labels = classes
    
    # Print the predictions
    print('Predictions:')
    for prob, label in zip(probs, labels):
        print(f"{label}: {prob:.4f}")
