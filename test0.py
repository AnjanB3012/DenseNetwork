import Model
import Layer
import ActivationFunctions

intLayers = [
    Layer.Layer(4,ActivationFunctions.ActivationFunctions().relu),
    Layer.Layer(20,ActivationFunctions.ActivationFunctions().sigmoid)
]
tempModel = Model.Model(2,intermediateLayers=intLayers)
values = tempModel.computeData([1,2])
print(values)
tempModel.trainModel([1,2],[5])
values = tempModel.computeData([1,2])
print(values)