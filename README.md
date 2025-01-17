Dense Network from Scratch
This project implements a machine learning model from scratch using a fully connected dense network. The implementation includes essential components such as nodes, layers, activation functions, forward propagation, and backpropagation without using external deep learning frameworks like TensorFlow or PyTorch.

Features
Custom Activation Functions: Implements ReLU, Sigmoid, Tanh, Step, Linear, and LeakyReLU.
Layered Architecture: Supports multiple hidden layers with flexible activation functions.
Backpropagation: Trains the network using gradient descent.
Customizable Model: Users can define their own layer configurations.
File Structure
ActivationFunctions.py - Implements various activation functions used in the model.
Node.py - Defines the structure of a node (neuron) in the neural network.
Layer.py - Implements layers containing multiple nodes and activation functions.
Model.py - Defines the structure of the full neural network, including forward propagation and backpropagation.
test0.py - A sample test script to create and train a neural network.
How It Works
Define the Network

Specify the number of input nodes.
Create intermediate layers with chosen activation functions.
Forward Propagation

Data flows through the network, computing values at each layer.
Backpropagation

The model updates weights using gradient descent to minimize errors.
