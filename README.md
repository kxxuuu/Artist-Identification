# Artist-Identification
Automated Artist Identification Using Deep Learning and Transfer Learning Techniques

## Project Overview
This project aims to develop a neural network model that can accurately identify artists based on their artwork. By leveraging deep learning and transfer learning techniques, we use a pre-trained ResNet50 model to classify artworks from a diverse dataset of artists. The project addresses challenges such as data imbalance, high dimensionality, and the subjectivity inherent in traditional art analysis.


## Data Collection
The dataset consists of images of artwork labeled by artists, collected from Kaggle's "Best Artworks of All Time" dataset. It includes metadata for each artist, such as the number of artworks and other relevant information.

### Components
artists.csv: Contains metadata about the artists.
images.zip: Full-sized images of artworks.
resized.zip: Resized images for quicker processing.


## Preprocessing
Preprocessing steps include:
Resizing: All images are resized to 224x224 pixels.
Normalization: Pixel values are normalized to the [0, 1] range.
Data Augmentation: Techniques such as horizontal and vertical flipping, and shearing are used to enhance the dataset and mitigate class imbalance.
Class Weights: Calculated to address the class imbalance issue, ensuring fairness and accuracy.


## Model Architecture
The architecture is based on the ResNet50 model, pre-trained on the ImageNet dataset. Key components include:

Global Average Pooling Layer: Reduces the feature map size.
Dense Layers: Two dense layers with Dropout and Batch Normalization for regularization.
Output Layer: A softmax activation function provides a probability distribution over the artist classes.


## Training Process
The training process includes two phases:

Initial Training: All layers are unfrozen for 10 epochs with a learning rate of 0.0001.
Fine-Tuning: Up to 50 epochs, with the first 50 layers of ResNet50 trainable, using callbacks for EarlyStopping and ReduceLROnPlateau.
