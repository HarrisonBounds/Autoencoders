# 1. AutoEncoder Implementation

## Describe your dataset and the steps that you used to create it
The original dataset consists of many images of emojis that are of size 256x256. I started by using the datasets library to download the emoji dataset from huggin face. After this, dataset needed to be augmented and split into training, validation, and test sets. The first augmentation step was to gather a subset of a the data based on the text in the label. We chose to use the text "face", which includes grinning face, winking face, etc. The gave us ~200 images to work with in total. We then split this set into training, validation, and test with a 60, 20, 20 split respectively using the train_test_split library from sklearn. The next augmentation that needed to be done was to create more data for each split. We created an augment_data function that applied PyTorch transformations to the data that randomly flipped the images, resized them to 3, 64, 64, and converted the images to tensors. We did this 5 times for each set that multiplying the number of images by 5. 

<center>
Original train set: 122, Augmented train set: 610
Original valid set: 41, Augmented valid set: 205
Original test set: 41, Augmented test set: 205
</center>

Finally, we created a loader for each set to easilt iterate through the images when passing them to our model.

## Provide a summary of your architecture
The architecture consists of 5 layers for the encoder and 5 layers for the decoder. We use a "hourglass" architecutre, as do most autoencoders. The first layer is the number of channels (3, RGB) to 128. We use Convolutional Layers to reduce the dimensions to a 2x2 using a kernel size of 3, stride of 2, and padding of 1 for each layer to eventually decrease the image to a 2x2. Then, that output is ran through the decoder, which uses `ConvTranspose2d` which is similar to the inverse of Conv2d. This layer expands the encoded input back into an output that follow the dimensions of the batch size, channels, width, and height. We then utilize these in the forward method of the AE model.

## Discuss and explain your design choices
At first, we were using linear layers, which are not as well suited for image inputs, so switching to Conv2d was essential. We started with an arbitrary number of layers and made sure to match the encoder to the decoder. The main challenge was making sure that input image size was matched by the output size. This took some trial and error, as well as calculating the kernel size, stride, and padding with this formula:

<center>
Output size = ((Input size + 2 * Padding - Kernel size) / Stride) + 1
</center>

For our training script, we made the mistake of making the learning rate too high which caused our model to not learn very well. After dropping the learning rate, the loss decreased quite nicely. Increasing the number of epochs also helped with decreasing the loss, as we found our model needed more time to reach a local minimum. 

## List Hyperparameters used in the model

Hyperparameters:
- Layer Type
- Number of input/ouput channels per layer
- Number of total layers
- Batch size
- Learning Rate
- Resize 
- Transformations
- Loss Function
- Weight Decay
- Optimizer
- Epochs

## Plot learning curves for training and validation loss as a function of training epochs

**Training**
<center>
<img src="loss_plots/training/loss_vs_iterations.png" alt="Alt text" width="300" height="250">
</center>

**Validation**

## Provide the final average error of your autoencoder on your test set


## Provide a side-by-side example of 5 input and output images
<center>
<img src="output_plots/training/input_vs_output.png" alt="Alt text" width="500" height="250">
</center>

## Discuss any decisions or observations that you find relevant

# 2. Separate dataset into 2 or more classes
## Describe how you separated your dataset into classes
The original dataset consisted of 256x256 emojis (images) with associated text which describes the image. We sorted through the dataset and sampled emojis which had the word "face" or "fist" appear in their description. After storing these emojis, we separated them into two classes based on the word "face" or "fist". All emojis corresponding to "fist" were assigned label 1, and all other face emojis were assigned label zero. After zipping and merging the images and their labels, I shuffled and split the data into training, validation and test sets. Additional augmentation was necesasay to increase the size of the dataset, since the "fist" emojis were very few in comparison to the "face" emojis.

## Describe your classification technique and hyper parameters
The classification technique involved using the output of the encoder network, flattening it into a single layer, and then use 2 `nn.Linear` layers along with ReLU and Dropout regularization. The last linear layer's ouput is the class logits. Cross entropy is used as a loss function for this classification task. The hyper parameters used are:
- Layer Type: nn.Linear (excluding the encoding layers)
- Number of input/ouput channels per layer: (32, 128) -> (128, 2)
- Number of total layers: 3 (excluding the encoding layers) 
- Batch size: 32
- Learning Rate: 0.001
- Resize: 64, 64
- Transformations: Resize, RandomHorizontalFlip
- Loss Function: Cross Entropy
- Weight Decay: 1e-5
- Optimizer: Adam
- Epochs: 50

## Plot learning curves for training and validation loss for MSE and classification accuracy
<center>
<img src="loss_plots/classifier/res.png">
</center>

## Provide a side by side example of 5 input and output images
<center>
<img src="output_plots/classifier/res.png">
</center>

## Discuss how incorporating classification as an auxillary task impact the performance of your autoencoder
By incorporating the classification as an auxillary task, the autoencoder seems to get better at extracting important features from the model bottleneck (encoder output). The classification loss makes it factor in the decision of predicting the correct label, while adjusting the weights involved in downsampling the image, thereby affecting the entire encoding decoding pipeline. However, in our case the model seems to overfit for the classification task, I believe this is due to the lack of diverse samples, especially for the fist set of images, but I still believe the performance should improve with the auxillary classification task.

## Speculate why the performance changed and recommend (but do not implement) an experiment to confirm or reject your speculation
The performance changed because the classification loss is added as a lambda factor into the total loss of the model. The weight given by the model to the classification task is a variable that can be changed using the lambda variable. This can serve as an important experiment to verify the model performance, as testing the model performance at different values of lambda, can serve as a metric to determine how the model weighs the classification and reconstruction tasks, and how one can affect the other.

# 3. Attribute composition with vector arithmetic

## Specify which attribute you selected, the vector arithmetic applied and the resulting image(s)

## Provide a qualitative evaluation of your composite image

## Discuss ways to improve the quality of your generated image
