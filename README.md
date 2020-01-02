# Handwritten-digit-classification-using-ML-Algorithms
1. PERCEPTRONS

Perceptron is an algorithm for supervised learning of binary classifiers. A perceptron takes several binary inputs and produces a single binary output. Here we learn to classify the handwritten digits in the MNIST dataset. The MNIST data set is divided into training set consisting of 60,000 examples and testing set consisting of 10,000 examples. All digit images have been size-normalized and centered in a fixed size image of 28 x 28 pixels. In the dataset each pixel of the image is represented by a value between 0 and 255, where 0 is black, 255 is white and anything in between is a different shade of grey. We train 10 perceptrons, whose targets correspond to the 10 digits from 0 to 9. Each perceptron will have 785 inputs including the bias input +1and 1 output.
	First , we perform preprocessing where each input values that are in range from 0 to 255 are scaled to a value between 0 and 1. This helps the weights from getting too large. Perceptrons are trained with three different learning rates 0.01,0.1,1.0. For each learning rate we choose random weights between -0.05 and 0.05 initially. Perform 50 epochs for each learning rate, Accuracy for both training and test data is computed before getting into weight updation (ie. at epoch 0). For each training example we compute the summation of inputs times the weights for all the 10 perceptrons. Output y is 1 when that summation is greater than 0 or the output y is 0 when the summation is less or equal to 0. We update the weights of all the perceptrons using perceptron learning algorithm.
wi ←wi + Δwi
where
Δwi =η (t^k − y^k )xi^k   
After each epoch, accuracy on training and test set is computed without changing the weights. The output with the highest value of sum of inputs times the weights is the prediction for that example. 
Confusion matrix is done for the test data. At the end we plot accuracy vs epoch graph for all the three learning rates.

2.NEURAL NETWORKS

A two-layer neural network with one hidden layer is implemented to perform the handwritten digit recognition task. We use MNIST dataset, this data set is divided into training set consisting of 60,000 examples and testing set consisting of 10,000 examples. All the values in the dataset are scaled to values between 0 and 1. The neural network takes 784 inputs, n hidden units and 10 output units. Every input unit connects to every hidden unit and every hidden unit connects to every output unit. Every hidden and output unit has a weighted connection from the bias unit.

Random numbers with small values between -0.5 and 0.5 are generated for the weights between hidden and input layer, weights between output and hidden layer. Output unit corresponds to 10 digits, target value is set to 0.9 and the rest as 0.1. Learning rate is set to 0.1. We propagate the example forward from input to the output class. The class with highest activated output unit is regarded as the predicted class for an example. The hidden and output units use the sigmoid function. Back-propagation with stochastic gradient descent is used to train the network i.e. for weight updation.

Experiment 1

The number of hidden units are varied in this experiment. For each value of n=10,20,100, we train the network on the 60000 training examples. Momentum is set to 0.9. After each epoch, accuracy of both the test and train data are calculated without changing the weights and are plotted at the end. The network is trained for 50 epochs.

Experiment 2

The value of momentum is varied in this experiment. Number of hidden units is fixed to 100 and momentum values are changed to 0.0,0.5,0.9,1.0. We train the network on the 60000 training examples using these momentum values. After each epoch, accuracy of both the test and train data are calculated without changing the weights and are plotted at the end. The network is trained for 50 epochs. Confusion matrix is done for the test data.

Experiment 3

The size of the training examples is varied in this experiment. Number of hidden units is fixed to 100 and momentum is fixed to 0.9. We train the network on one quarter and one half of the training examples. After each epoch, accuracy of both the test and train data are calculated without changing the weights and are plotted at the end. The network is trained for 50 epochs. Confusion matrix is done for the test data.
