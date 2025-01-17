import Node

class Layer:
    def __init__(self, numberOfNodes, activationFunction):
        self.__Nodes = []
        self.__activationFunction = activationFunction
        for i in range(numberOfNodes):
            self.__Nodes.append(Node.Node(activationFunction=self.__activationFunction))
        self.__nextLayer = None
        self.__backLayer = None
        self.__storeValues = [0,0,0,0,0,0]
    
    def setNextLayer(self, nextLayer):
        self.__nextLayer = nextLayer
    
    def getActivationFunction(self):
        return self.__activationFunction

    def setBackLayer(self, backLayer):
        self.__backLayer = backLayer

    def computeValue(self, backValues):
        tempDict = {}
        for node in self.__Nodes:
            tempDict[node] = node.computeValue(backValues)
        return tempDict
    
    def getNextLayer(self):
        return self.__nextLayer
    
    def getBackLayer(self):
        return self.__backLayer
    
    def getNodes(self):
        return self.__Nodes
    
    def forceSetData(self, inputData):
        tempDict = {}
        for i in range(len(self.__Nodes)):
            tempDict[self.__Nodes[i]]=self.__Nodes[i].computeValue({},inputData[i])
        return tempDict

    def backpropagateLayer(self, errorSignals, learningRate):
        newErrorSignals = {}
        for node in self.__Nodes:
            newErrors = node.backPropagation(errorSignals[node], learningRate)
            for key in newErrors:
                if key in newErrorSignals:
                    newErrorSignals[key] += newErrors[key]
                else:
                    newErrorSignals[key] = newErrors[key]
        return newErrorSignals