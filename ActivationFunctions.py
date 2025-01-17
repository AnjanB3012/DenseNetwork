import math
class ActivationFunctions:
    def __init__(self):
        pass
    def linear(self, inputValue, train=False):
        if train:
            return 1
        return inputValue
    def stepThreshold(self, inputValue, train=False):
        if train:
            return 0
        if(inputValue>0):
            return 1
        return 0
    def sigmoid(self, inputValue, train=False):
        if train:
            return (1/(1+((math.e)**(-inputValue)))) * (1 - (1/(1+((math.e)**(-inputValue)))))
        return 1/(1+((math.e)**(-inputValue)))
    def tanh(self, inputValue, train=False):
        if train:
            return 1 - (((math.e)**inputValue)-((math.e)**(-inputValue)))/(((math.e)**inputValue)+((math.e)**(-inputValue)))**2 
        return (((math.e)**inputValue)-((math.e)**(-inputValue)))/(((math.e)**inputValue)+((math.e)**(-inputValue)))
    def relu(self, inputValue, train=False):
        if train:
            return 1 if inputValue > 0 else 0
        return max(0,inputValue)
    def leakyReLU(self, inputValue, train=False):
        if train:
            return 1 if inputValue > 0 else 0.01
        if(inputValue>0):
            return inputValue
        return 0.01*inputValue