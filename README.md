# Udacity Project 2: My Own Image Classifier

This project is part of the Udacity AI Programming with Python Nanodegree, focusing on developing an image classifier using deep learning techniques with PyTorch. The model is trained to classify images into categories using the [102 Flower Categories dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

## Repository Structure


### Key Files:
- **train.py**: Script to train the image classifier.
- **predict.py**: Script to use the trained model for predictions.
- **modelfuncs.py**: Contains model-building, checkpoint saving, and loading functions.
- **imageFuncs.py**: Contains image processing and transformation functions.

## Project Overview

This project is divided into two parts:

1. **Part 1: Training the Model**
   - Preprocesses the dataset, builds the model, trains it, and saves a checkpoint of the trained model.
2. **Part 2: Prediction Script**
   - Implements a command-line interface to load a trained model and predict the class of new images.

## Prerequisites

Ensure you have the following installed:
- Python 3.x
- [PyTorch](https://pytorch.org/)
- torchvision
- PIL (Python Imaging Library)
- matplotlib
- numpy

You can install these dependencies using:

```bash
pip install torch torchvision pillow matplotlib numpy
 
Training

To train the model, run the following command:
python train.py --data_dir <path_to_data> --save_dir <path_to_save_checkpoint> --arch <model_name> --learning_rate <lr> --hidden_units <units> --epochs <epochs> --gpu

 Exemple
python train.py --data_dir ./flowers --save_dir ./checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 20 --gpu

--data_dir: Path to the dataset (flowers directory).
--save_dir: Directory where the checkpoint will be saved.
--arch: Model architecture to use (vgg16, resnet50).
--learning_rate: Learning rate for training.
--hidden_units: Number of hidden units in the classifier.
--epochs: Number of training epochs.
--gpu: Use GPU for training if available.
Making Predictions
To make predictions using the trained model, run:
python predict.py --image_path <image_path> --checkpoint <checkpoint_path> --top_k <K> --category_names <json_file> --gpu
exemple
python predict.py --image_path ./flowers/test/1/image_06752.jpg --checkpoint ./checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu

--image_path: Path to the image to be classified.
--checkpoint: Path to the model checkpoint.
--top_k: Number of top predictions to display.
--category_names: JSON file mapping category labels to actual flower names.
--gpu: Use GPU for prediction if available.
Dataset
The project uses the 102 Flower Categories dataset from Oxford, which contains images of 102 flower species. The dataset is split into training, validation, and test sets:

train/: Training data.
valid/: Validation data.
test/: Test data.
Label Mapping
The cat_to_name.json file contains the mapping from category labels (integers) to flower names, allowing predictions to display actual names.

Model and Checkpoint
The model used for training is based on a pre-trained network (e.g., VGG16 or ResNet50) with a custom classifier. The checkpoint saves the model architecture, state dictionary, class-to-index mapping, optimizer state, and other training parameters.

Acknowledgements
This project is part of the Udacity AI Programming with Python Nanodegree.
The dataset used is the 102 Flower Categories dataset provided by the University of Oxford.


