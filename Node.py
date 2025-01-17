import random
class Node:
    def __init__(self, activationFunction):
        self.__bias = 0
        self.__backWeights = {}
        self.__actFunc = activationFunction
        self.__storeValues = [0,0,0,0,0,0]
        
    def connectNodeBack(self, backNode):
        self.__backWeights[backNode] = random.randint(1,3)
    
    def computeValue(self, backValues, secondaryInput=0):
        tempCompute = self.__bias + secondaryInput
        for key in backValues:
            tempCompute+=backValues[key]*self.__backWeights[key]
        self.__storeValues[0] = tempCompute
        self.__storeValues[1] = backValues
        tempCompute = self.__actFunc(tempCompute)
        return tempCompute
    
    def updateWeight(self, newWeights):
        self.__backWeights = newWeights

    def updateBias(self, newBias):
        self.__bias = newBias
    
    def getData(self):
        return [self.__backWeights,self.__storeValues]
    
    def getActFunc(self):
        return self.__actFunc
    
    def backPropagation(self, errorSignal, learningRate=0.01):
        activationDerivative = self.__actFunc(self.__storeValues[0], train=True)

        delta = errorSignal * activationDerivative

        newWeights = {}
        for node, activation in self.__storeValues[1].items():
            newWeights[node] = self.__backWeights[node] - learningRate * delta * activation

        newBias = self.__bias - learningRate * delta
        self.__backWeights = newWeights
        self.__bias = newBias

        newErrorSignals = {}
        for node, weight in self.__backWeights.items():
            if node in newErrorSignals:
                newErrorSignals[node] += delta * weight
            else:
                newErrorSignals[node] = delta * weight
        return newErrorSignals