import Layer
import ActivationFunctions

class Model:
    def __init__(self, numberOfParameters, intermediateLayers):
        self.__front = Layer.Layer(numberOfParameters, ActivationFunctions.ActivationFunctions().linear)
        self.__output = Layer.Layer(1, ActivationFunctions.ActivationFunctions().linear)
        currLay = self.__front
        for i in range(len(intermediateLayers)):
            currLay.setNextLayer(intermediateLayers[i])
            self.__connectNodes(currLay)
            currLay = intermediateLayers[i]
        currLay.setNextLayer(self.__output)
        self.__connectNodes(currLay)
    
    def __connectNodes(self,layer):
        layer1Nodes = layer.getNodes()
        layer2Nodes = layer.getNextLayer().getNodes()
        for node2 in layer2Nodes:
            for node1 in layer1Nodes:
                node2.connectNodeBack(node1)

    def computeData(self,inputData):
        nextDict = self.__front.forceSetData(inputData=inputData)
        currLay = self.__front.getNextLayer()
        while(currLay!=None):
            nextDict = currLay.computeValue(nextDict)
            currLay = currLay.getNextLayer()
        return nextDict
    
    def trainModel(self, inputData, expectedOutputs, learningRate=0.01, epochs=100):
        for i in range(epochs):
            computedOutputs = self.computeData(inputData)

            outputNodes = self.__output.getNodes()
            errorSignals = {}
            for node, expected in zip(outputNodes, expectedOutputs):
                errorSignals[node] = 2 * (computedOutputs[node] - expected)

            currLayer = self.__output
            while currLayer is not None:
                errorSignals = currLayer.backpropagateLayer(errorSignals, learningRate)
                currLayer = currLayer.getBackLayer()