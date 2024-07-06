**My personally written neural network.**

**Example of how to run the neural network on any dataset**
(with any number of numeric and categorical features as well as with any number of labels of numeric and categorical types.
String categorical values for features and labels are possible):

data = readCSVtoArr2D('datasets/data.csv', \<RowToBeRemoved>) #\<RowToBeRemoved> is optional. It can be used to remove column names.<br>
dataStruct = NeuralNetwork.getDataStruct(data, ['num','cat','cat','num','cat','cat','cat','cat','cat','labelCat'])<br>
neuralNetwork = NeuralNetwork(dataStruct)<br>
neuralNetwork.train(data)<br>
neuralNetwork.predict(removeLabels(data, dataStruct["labelCols"]))<br>