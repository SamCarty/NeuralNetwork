# Custom Neural Network (NN) for the MNIST Dataset in MATLAB

![MNIST dataset sample](https://user-images.githubusercontent.com/22345452/82841322-8419f780-9ecd-11ea-9cdd-57659aeff07d.png)

## What is MNIST?
View on [Wikipedia](https://en.wikipedia.org/wiki/MNIST_database):

> The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning."

## Implementation
### Neural Network
In the implementation for MNIST, the NN has 784 input nodes (one for each pixel), 14 hidden layer nodes and a 10 out layer nodes (each representing a digit between 0-9):

![Neural Network - input layer, hidden layer and output layer](https://user-images.githubusercontent.com/22345452/82841410-d65b1880-9ecd-11ea-955d-e4214817b66b.png)

### Hyperparameters
A learning rate of 0.3 proved optimal for the MNIST dataset, providing a balance of performance and accuracy:
![image](https://user-images.githubusercontent.com/22345452/82842278-d27cc580-9ed0-11ea-8c33-da254a6d459b.png)
> Graph showing the epoch number versus the error rate for a training dataset of 7,000 items and a testing dataset of 3,000 items.

The sigmoid function was used as the activation function, allowing for a non-linear activation which is required for the MNIST problem. In the output layer, the softmax function is used to ensure the outputs of each node is a value between 0 and 1, providing a possibility indicator for each handwritten digit between 0-9.

Finally, the cross-entropy loss function is used for the NN. This is due to its compatibility with the 0-1 probability range of softmax. 

## Results
When trained originally using a simple XOR dataset, the neural network (NN) achieved an error rate of just 0.0039, showing that for 99.61% of values, the correct classification was obtained:

![image](https://user-images.githubusercontent.com/22345452/82841869-7cf3e900-9ecf-11ea-83ee-94758c38c7b4.png)

![image](https://user-images.githubusercontent.com/22345452/82841871-7f564300-9ecf-11ea-844c-7f4291403630.png)

When transferred to the MNIST dataset, an error rate of 0.0375 was observed over 3 epochs, meaning that 96.25% of handwritten digits were identified correctly. 

It is probable that more epochs would reduce the error rate further, however the execution time on each epoch was around 15 minutes when exposed to the full dataset of 70,000 items. Therefore, due to time constraints, this was the limit of experimentation.

## Further Research
This NN displayed great results considering the network has only one hidden layer. Adding additional hidden layers would likely reduce the error rate further, and the inclusion of more sophisticated dimensionality reduction measures would improve the training speed of the model. 

Finally, [convolutional NNs](http://yann.lecun.com/exdb/mnist/) seem to perform particularly well on the MNIST problem by breaking the images down into smaller regions, reducing the number of dimensions in the data.
