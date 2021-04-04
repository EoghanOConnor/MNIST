#########################################################################
#
# Author : 	Eoghan O'Connor 
#
# File Name: 	MNIST 
#
# Description:  A 2-layered convolutional layer is created,
#				1st layer 5x5x32 followed by relu activation 
#				and a 3x3 maxpooling. 
#				The is the input for 3x3x10 hidden Conv2D layer
#				followed by a dropout,2x2 maxpooling layer and a
#				relu activation.
#				
#				This is flattened and inputted into 2-layer dense 
#				layer.
#				A dropout function is used between the layers.
#				The output of this is given to a softmax activtion 
#				where the outcome is used for
#				backpropagtion and one hot encoding the labels.
#
#
##########################################################################




#Libraries imported
import numpy as np
import matplotlib.pyplot as plt

# Initialise the random number generators to get exactly the same 
# results from each run.  This has to be done before any other 
# network initialisation.  

np.random.seed(1)                # Initialise system RNG.

import tensorflow
tensorflow.random.set_seed(2)    # and the seed of the Tensorflow backend.



# Import the relevant Keras library modules.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense          # Fully-connected layer
from tensorflow.keras.layers import Conv2D         # 2-d Convolutional layer
from tensorflow.keras.layers import MaxPooling2D   # 2-d Max-pooling layer
from tensorflow.keras.layers import Flatten        # Converts 2-d layer to 1-d layer
from tensorflow.keras.layers import Activation     # Nonlinearities
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical


# Load up the MNIST dataset from repository


from tensorflow.keras.datasets import mnist


# Dividing the 60,000 28 x 28 greyscale images into training images and testing images
# The same is done with the labels

(training_inputs, training_labels), (testing_inputs, testing_labels) = mnist.load_data()

print(training_inputs.shape, training_inputs.dtype, testing_inputs.shape, testing_inputs.dtype)

# The images are normalized to float32 values
#The labels are one hot encoded using the categorical function

training_images = (training_inputs.astype('float32')/255)[:,:,:,np.newaxis]  # Normalised float32 4-tensor.

categorical_training_outputs = to_categorical(training_labels)

testing_images = (testing_inputs.astype('float32')/255)[:,:,:,np.newaxis]

categorical_testing_outputs = to_categorical(testing_labels)

print(training_images.shape,training_images.dtype)
print(testing_images.shape,testing_images.dtype)
print(categorical_training_outputs.shape, training_labels.shape)
print(categorical_testing_outputs.shape, testing_labels.shape)

plt.figure(figsize=(14,4))
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(training_images[i,:,:,0],cmap='gray')
    plt.title(str(training_labels[i]))
    plt.axis('off')




# The model is created using sequential
# The convolutional layer uses a 5x5 sampling window and has 32 slices (5x5x32)
# Edges are padded with copies of the input data
# Stride on the convolutional layer is implicitly 1
# A Relu activation function is applied after this
#
# The maxpooling layer uses a 3x3 sampling window, with a stride of 2 in x and y axis
#
# The 2nd convolitional layer uses a 3x3 sampling window and has 10 slices (3x3x10)
# Edges are padded with copies of the input data
# Stride on the convolutional layer is implicitly 1
# A Relu activation function is applied after this
#
# A dropout layer of .2 is invoked after this prevent overfitting of the model.
#
# The maxpooling layer uses a 2x2 sampling window, with a stride of 2 in x and y axis
#
# The network is then flattened to 1d array using the flatten function
# This is a hidden layer which provides the input to the dense layer
# A dense layer of 128 neurons is applied with a relu activation function.
# Another dropout layer of .2 is invoked again.
#
# Finally the output layer with a dense layer of 10 neurons and a softmax activation function
# applied with is used for classification and back propagation 


model = Sequential([
            Conv2D(32, kernel_size=5, padding='same', input_shape=training_images.shape[1:]),
            Activation('relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Conv2D(10, kernel_size=3, padding='same', input_shape=training_images.shape[1:]),
            Activation('relu'),
            Dropout(0.4),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Flatten(),
            Dense(128),
            Activation('relu'),
            Dropout(0.2),
            Dense(10),
            Activation('softmax')])


# Reviewing the model
print("The Keras network model")
model.summary()


# The model is compiled using the categorical_crossentropy , this catagories
# the inputs from the softmax as a percentage of each of the labels
# the highest value being the the most likely digit for that image.
#
# Adam optimizer is used here for its adapative gradient algorithm.

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# Using the fit function we attempt to generate a model that fits
# the training set.
#
# This is done by giving the function the images and labels.
# The number of epochs is limited to 11.
# The batch size for each training cycle is set to 32
# The data is suffled.
# A split of the data is 15% between training and validation.
# Verbose displays the training as it happens.

history = model.fit(training_images, categorical_training_outputs,
                    epochs=11, 
                    batch_size=32, 
                    shuffle=True, 
                    validation_split=0.15,
                    verbose=2)



# Graphing the training and validation loss for each epoch

plt.figure('Training and Validation Losses per epoch', figsize=(8,8))

plt.plot(history.history['loss'],label='training') # Training data error per epoch.

plt.plot(history.history['val_loss'],label='validation') # Validation error per ep.

plt.grid(True)

plt.legend()

plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()

# Comparing the model with unseen/used dataset

print("Performance of network on testing set:")
test_loss,test_acc = model.evaluate(testing_images,categorical_testing_outputs)
print("Accuracy on testing data: {:6.2f}%".format(test_acc*100))
print("Test error (loss):        {:8.4f}".format(test_loss))


# Reporting the results of the model performance.

print("Performance of network:")
print("Accuracy on training data:   {:6.2f}%".format(history.history['accuracy'][-1]*100))
print("Accuracy on validation data: {:6.2f}%".format(history.history['val_accuracy'][-1]*100))
print("Accuracy on testing data:    {:6.2f}%".format(test_acc*100))
