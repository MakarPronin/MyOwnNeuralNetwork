import numpy as np
import pandas
import copy
from datetime import datetime

#GENERAL FUNCTIONS THAT DO NOT RELATE TO NEURAL NETWORK
def readCSVtoArr2D(dir, header=None, sep=None):
    arr2D = pandas.read_csv(dir, header=header, sep=sep, engine='python').dropna(axis=1).to_numpy(dtype=object)
    return arr2D

def removeLabels(data, labelCols):
    return np.delete(data, labelCols, axis=1)

#NEURAL NETWORK
class NeuralNetwork():
    def __init__(self, dataStruct, hiddenStruct=[64], reg=0.2, stepSize=1, initWeights=None, demo=False, textFileName=None):
        def calcInputLayerSize():
            size = 0
            for col in self.dataStruct["colsDescr"]:
                if (col["type"] == "num"):
                    size += 1
                if (col["type"] == "cat"):
                    size += col["oneHotLen"]

            return size
        
        def calcOutputLayerSize():
            sum = 0
            for col in self.dataStruct["labelCols"]:
                if self.dataStruct["colsDescr"][col]["type"] == "labelNum":
                    sum += 1
                if self.dataStruct["colsDescr"][col]["type"] == "labelCat":
                    sum += self.dataStruct["colsDescr"][col]["oneHotLen"]
            return sum
            
        def initWeightMatrix(initWeights=None):
            weightMatrix = [None]

            if initWeights != None:
                for layer in initWeights:
                    if not layer is None:
                        weightMatrix.append(np.array(layer))
            else:
                for layer in range(1, len(self.struct)):
                    numNeuronsInPrevLayer = self.struct[layer-1]
                    numNeuronsInCurLayer = self.struct[layer]

                    weightMatrix.append(np.random.normal(loc=0, scale=1, size=(numNeuronsInCurLayer, numNeuronsInPrevLayer+1)))

            self.weights = weightMatrix
        
        def outputInfoAboutNN():
            if textFileName is None:
                self.textFile = 'NNAndBackpropInfo'+datetime.now().strftime("_%d_%m_%Y__%H_%M_%S")+".txt"
            else:
                self.textFile = textFileName + '.txt'
            with open(self.textFile, 'w') as file:
                file.write(f'Regularization parameter lambda={self.reg:.3f}\n\n')
                structure = self.struct
                structStr = ' '.join(map(str, structure))
                structStr = '[' + structStr + ']'
                file.write(f'Initializing the network with the following structure (number of neurons per layer): {structStr}\n\n')
                for layer in range(1, len(structure)):
                    file.write(f'Initial Theta{layer} (the weights of each neuron, including the bias weight, are stored in the rows):\n')
                    for row in self.weights[layer]:
                        file.write('\t')
                        for w in row:
                            file.write(f"{w:9.5f}")
                        file.write('\n')
                    file.write('\n')
                file.write('\n')

        self.dataStruct = dataStruct
        self.struct = [calcInputLayerSize()] + hiddenStruct + [calcOutputLayerSize()]
        self.reg = reg
        self.stepSize = stepSize
        self.demo = demo
        initWeightMatrix(initWeights)
        if self.demo:
            outputInfoAboutNN()

    def normalizeSamples(self, samples):
        def normalizeSample(sample):
            if (len(sample) == len(self.dataStruct["colsDescr"])):
                withLabel = True
            else:
                withLabel = False

            normSampleX = np.array([])
            normSampleY = np.array([])

            labelMissed = 0
            for col in range(len(self.dataStruct["colsDescr"])):
                if self.dataStruct["colsDescr"][col]["type"] == "num":
                    minMaxDiff = float(self.dataStruct["colsDescr"][col]["max"] - self.dataStruct["colsDescr"][col]["min"]) 
                    if minMaxDiff == 0:
                        normSampleX = np.append(normSampleX, float(sample[col-labelMissed]))
                    else:
                        normSampleX = np.append(normSampleX, float(sample[col-labelMissed] - self.dataStruct["colsDescr"][col]["min"])/minMaxDiff)

                if self.dataStruct["colsDescr"][col]["type"] == "cat":
                    oneHotVector = np.zeros(self.dataStruct["colsDescr"][col]["oneHotLen"])
                    oneHotVector[self.dataStruct["colsDescr"][col]["oneHot"][sample[col-labelMissed]]] = 1
                    normSampleX = np.append(normSampleX, oneHotVector)

                if self.dataStruct["colsDescr"][col]["type"] == "labelCat":
                    if (withLabel):
                        oneHotVector = np.zeros(self.dataStruct["colsDescr"][col]["oneHotLen"])
                        oneHotVector[self.dataStruct["colsDescr"][col]["oneHot"][sample[col-labelMissed]]] = 1
                        normSampleY = np.append(normSampleY, oneHotVector)
                    else:
                        labelMissed += 1

                if self.dataStruct["colsDescr"][col]["type"] == "labelNum":
                    if (withLabel):
                        minMaxDiff = float(self.dataStruct["colsDescr"][col]["max"] - self.dataStruct["colsDescr"][col]["min"])
                        if minMaxDiff == 0:
                            normSampleY = np.append(normSampleY, float(sample[col-labelMissed]))
                        else: 
                            normSampleY = np.append(normSampleY, float(sample[col-labelMissed] - self.dataStruct["colsDescr"][col]["min"])/minMaxDiff)
                    else:
                        labelMissed += 1

            return {"x": normSampleX, "y": normSampleY}

        X = []
        Y = []
        for sample in samples:
            result = normalizeSample(sample)
            X.append(result['x'])
            Y.append(result['y'])
        
        return {"x": np.array(X), "y": np.array(Y)}

    def predictVector(self, inputVectors, withActivations=False):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if (withActivations):
            activations = []
            zValues = []

        onesColumn = np.ones((len(inputVectors), 1))
        curActivationsArr = np.hstack((onesColumn, inputVectors))
        if (withActivations):
            activations.append(curActivationsArr)
            zValues.append(None)

        for layer in range(1, len(self.struct) - 1):
            curZArr = curActivationsArr @ self.weights[layer].T
            curActivationsArr = sigmoid(curZArr)
            onesColumn = np.ones((curActivationsArr.shape[0], 1))
            curActivationsArr = np.hstack((onesColumn, curActivationsArr))
            if (withActivations):
                activations.append(curActivationsArr)
                zValues.append(curZArr)

        curZArr = curActivationsArr @ self.weights[len(self.struct) - 1].T
        curActivationsArr = sigmoid(curZArr)
        if (withActivations):
            activations.append(curActivationsArr)
            zValues.append(curZArr)

        if (withActivations):
            return {"activations": activations, "zValues": zValues}
        else:
            return curActivationsArr

    def calcJ(self, outVects, expOutVects, withReg=True):        
        def squaredWeightsSum():
            sum = 0
            for layer in range(1, len(self.struct)):
                noBiasesLayerMatrix = np.delete(self.weights[layer], 0, axis=1)
                sum += np.sum(np.square(noBiasesLayerMatrix))
            
            return sum

        outVects = np.array(outVects)
        expOutVects = np.array(expOutVects)

        sum = np.sum(-expOutVects*np.log(outVects)-(1-expOutVects)*np.log(1-outVects))
        
        if (withReg):
            regAddition = (self.reg * squaredWeightsSum())/2/len(outVects)
        else:
            regAddition = 0
        
        return sum/len(outVects) + regAddition
    
    def numGradAppr(self, data, epsilon, normalized=True):
        def outputNumApprInfo(gradientApprObj):
            with open(self.textFile, 'a') as file:
                file.write('\n\n\n--------------------------------------------\n')
                file.write(f"Numeric approximation of gradients with epsilon {epsilon} (DON'T CONFUSE THIS PART WITH ACTUAL GRADIENTS!!!)\n")
                for i in range(len(data)):
                    file.write(f'\tComputing gradients based on training instance {i+1}\n')
                    for layer in range(len(self.struct)-1, 0, -1):
                        file.write(f'\t\tGradients of Theta{layer} based on training instance {i+1}:\n')
                        for row in gradientApprObj["gradientAppr"][layer][i]:
                            file.write('\t\t')
                            for val in row:
                                file.write(f'{val:9.5f}')
                            file.write('\n')
                        file.write('\n')
                file.write('\tComputing the average (regularized) gradients:\n')
                for layer in range(1, len(self.struct)):
                    file.write(f'\t\tFinal regularized gradients of Theta{layer}:\n')
                    for row in gradientApprObj["finGradientsAppr"][layer]:
                        file.write('\t\t')
                        for val in row:
                            file.write(f'{val:9.5f}')
                        file.write('\n')
                    file.write('\n')
                file.write('\n')

        if (not normalized):
            normSamples = self.normalizeSamples(data)
        else:
            X = []
            Y = []
            for sample in data:
                X.append([item for i, item in enumerate(sample) if not i in self.dataStruct["labelCols"]])
                Y.append([item for i, item in enumerate(sample) if i in self.dataStruct["labelCols"]])
            normSamples = {"x": np.array(X), "y": np.array(Y)}

        gradientAppr = [None]
        weightsOrig = copy.deepcopy(self.weights)
        for layer in range(1, len(self.weights)):
            gradientsLayer = []
            for row in range(len(self.weights[layer])):
                gradientsRow = []
                for col in range(len(self.weights[layer][row])):
                    weightsCopy0 = copy.deepcopy(weightsOrig)
                    weightsCopy1 = copy.deepcopy(weightsOrig)
                    weightsCopy0[layer][row][col] -= epsilon
                    weightsCopy1[layer][row][col] += epsilon
                    j0 = []
                    j1 = []

                    self.weights = weightsCopy0
                    outVect = self.predictVector(normSamples['x'])
                    for i in range(len(normSamples['x'])):
                        j0.append(self.calcJ([outVect[i]], [normSamples['y'][i]], False))

                    self.weights = weightsCopy1
                    outVect = self.predictVector(normSamples['x'])
                    for i in range(len(normSamples['x'])):
                        j1.append(self.calcJ([outVect[i]], [normSamples['y'][i]], False))

                    gradientsRow.append((np.array(j1)-np.array(j0))/(2*epsilon))
                gradientsLayer.append(gradientsRow)
            gradientsLayer = np.transpose(gradientsLayer, (2, 0, 1))
            gradientAppr.append(gradientsLayer)
        self.weights = weightsOrig

        finGradientsAppr = [None]
        for layerK in range(1, len(self.struct)):
            avrgGradients = np.sum(gradientAppr[layerK], axis=0) / len(normSamples['y'])
            weightsNoBiases = self.weights[layerK].copy()
            weightsNoBiases[:, 0] = 0
            avrgGradients += (self.reg * weightsNoBiases) / len(normSamples['y'])
            finGradientsAppr.append(avrgGradients)
            self.weights[layerK] -= self.stepSize * avrgGradients

        gradientApprObj = {"gradientAppr": gradientAppr, "finGradientsAppr": finGradientsAppr}

        if self.demo:
            outputNumApprInfo(gradientApprObj)

        return gradientApprObj

    def getDataStruct(dataIn, numCatlabelStruct):
        #information about columns of the dataset. {colsDescr: [{type: "num", max: Float, min: Float}, {type: "cat", oneHot: {"cat": 0, "dog": 1}, oneHotRev: {0: "cat", 1: "dog"}, oneHotLen: 2}, {type: "labelCat", oneHot: {"animal": 0, "bird": 1}, oneHotRev: {0: "bird", 1: "animal"}, oneHotLen: 2}], numerCols: [0], catCols: [1], labelCols: [2]}
        data = np.array(dataIn, dtype=object)
        dataStruct = {"colsDescr": [], "numerCols": [], "catCols": [], "labelCols": []}
        for col in range(len(numCatlabelStruct)):
            dataCol = {}
            type = numCatlabelStruct[col]
            dataCol["type"] = type
            if (type == "num"):
                dataStruct["numerCols"].append(col)
            if (type == "cat"):
                dataStruct["catCols"].append(col)
            if (type == "labelCat" or type == "labelNum"):
                dataStruct["labelCols"].append(col)

            if (type == "num" or type =="labelNum"):
                dataCol["min"] = data[:, col].min()
                dataCol["max"] = data[:, col].max()

            if (type == "cat" or type == "labelCat"):
                uniqueCategories = np.unique(data[:, col])
                oneHotInterpet = {}
                oneHotInterpetRev = {}
                for i in range(len(uniqueCategories)):
                    oneHotInterpet[uniqueCategories[i]] = i
                    oneHotInterpetRev[i] = uniqueCategories[i]
                dataCol["oneHot"] = oneHotInterpet
                dataCol["oneHotRev"] = oneHotInterpetRev
                dataCol["oneHotLen"] = len(uniqueCategories)

            dataStruct["colsDescr"].append(dataCol)

        return dataStruct

    def train(self, data, batchSize=0.1, numEpochs=500, replace=True, normalized=False):
        def backProp(normSamples, activations, returnDeltas=False):
            actvns = activations["activations"]
            deltasMatrix = []
            curDeltas = actvns[len(actvns) - 1] - normSamples['y']
            deltasMatrix.append(curDeltas)

            for layerK in range(len(actvns) - 2, 0, -1):
                weightsNoBiases = self.weights[layerK + 1][:, 1:]
                actvnsNoBiasNeurons = actvns[layerK][:, 1:]
                curDeltas = (curDeltas @ weightsNoBiases) * actvnsNoBiasNeurons*(1-actvnsNoBiasNeurons)
                deltasMatrix.append(curDeltas)

            deltasMatrix.append(None)

            deltasMatrix = list(reversed(deltasMatrix))

            gradients = [None]
            for layerK in range(1, len(actvns)):
                gradients.append(deltasMatrix[layerK][:, :, np.newaxis] * actvns[layerK-1][:,np.newaxis,:])

            if (returnDeltas):
                finGradients = [None]
            for layerK in range(1, len(actvns)):
                avrgGradients = np.sum(gradients[layerK], axis=0) / len(normSamples['y'])
                weightsNoBiases = self.weights[layerK].copy()
                weightsNoBiases[:, 0] = 0
                avrgGradients += (self.reg * weightsNoBiases) / len(normSamples['y'])
                if (returnDeltas):
                    finGradients.append(avrgGradients)
                self.weights[layerK] -= self.stepSize * avrgGradients

            if returnDeltas:
                #{"deltas": deltas[#layer][#sample][#neuron], "gradients": gradients[#layer][#sample][#neuron(L)][#neuron(L-1)], "finGradients": finGradients[#layer][#neuron(L)][#neuron(L-1)]}
                return {"deltas": deltasMatrix, "gradients": gradients, "finGradients": finGradients}
        
        def outputTrainDataInfo(normSamples):
            with open(self.textFile, 'a') as file:
                file.write('Training set\n')
                for sampleI in range(len(normSamples["x"])):
                    file.write(f'\tTraining instance {sampleI+1}\n')
                    file.write('\t\tx: [')
                    for val in normSamples["x"][sampleI]:
                        file.write(f'{val:9.5f}')
                    file.write(']\n')
                    file.write('\t\ty: [')
                    for val in normSamples["y"][sampleI]:
                        file.write(f'{val:9.5f}')
                    file.write(']\n')
                file.write('\n')

        def outputFeedForwardInfo(batch, actvns):
            with open(self.textFile, 'a') as file:
                file.write('--------------------------------------------\n')
                if (numEpochs > 1):
                    file.write(f'Epoch {epoch+1}')
                file.write('Computing the error/cost, J, of the network\n')
                for i in range(len(batch)):
                    file.write(f'\tProcessing training instance {i+1}\n')
                    file.write('\tForward propagating the input [')
                    for val in batch["x"][i]:
                        file.write(f'{val:9.5f}')
                    file.write(']\n')
                    for layer in range(len(self.struct)):
                        if (not actvns["zValues"][layer] is None):
                            file.write(f'\t\tz{layer+1}: [')
                            for neuron in range(len(actvns["zValues"][layer][i])):
                                file.write(f'{actvns["zValues"][layer][i][neuron]:9.5f}')
                            file.write(']\n')
                        file.write(f'\t\ta{layer+1}: [')
                        for neuron in range(len(actvns["activations"][layer][i])):
                            file.write(f'{actvns["activations"][layer][i][neuron]:9.5f}')
                        file.write(']\n')
                        file.write('\n')
                    file.write('\t\tf(x): [')
                    for outputNeuron in actvns["activations"][len(self.struct)-1][i]:
                        file.write(f'{outputNeuron:9.5f}')
                    file.write(']\n')
                    file.write(f'\tPredicted output for instance {i+1}: [')
                    for outputNeuron in actvns["activations"][len(self.struct)-1][i]:
                        file.write(f'{outputNeuron:9.5f}')
                    file.write(']\n')
                    file.write(f'\tExpected output for instance {i+1}: [')
                    for outputNeuron in batch['y'][i]:
                        file.write(f'{outputNeuron:9.5f}')
                    file.write(']\n')
                    file.write(f'\tCost, J, associated with instance {i+1}: {self.calcJ([actvns["activations"][len(self.struct)-1][i]], [batch["y"][i]], False):.3f}')
                    file.write('\n\n')
                file.write(f'Final (regularized) cost, J, based on the complete training set: {self.calcJ(actvns["activations"][len(self.struct)-1], batch["y"]):.5f}\n\n\n')

        def outputBackPropInfo(backPropResultObj, batch):
            deltas = backPropResultObj["deltas"]
            gradients = backPropResultObj["gradients"]
            finGradients = backPropResultObj["finGradients"]
            with open(self.textFile, 'a') as file:
                file.write('--------------------------------------------\n')
                file.write('Running backpropagation\n')
                for i in range(len(batch)):
                    file.write(f'\tComputing gradients based on training instance {i+1}\n')
                    for layer in range(len(self.struct)-1, 0, -1):
                        file.write(f'\t\tdelta{layer+1}: [')
                        for val in deltas[layer][i]:
                            file.write(f'{val:9.5f}')
                        file.write(']\n')
                    file.write('\n')
                    for layer in range(len(self.struct)-1, 0, -1):
                        file.write(f'\t\tGradients of Theta{layer} based on training instance {i+1}:\n')
                        for row in gradients[layer][i]:
                            file.write('\t\t')
                            for val in row:
                                file.write(f'{val:9.5f}')
                            file.write('\n')
                        file.write('\n')
                file.write('\tThe entire training set has been processes. Computing the average (regularized) gradients:\n')
                for layer in range(1, len(self.struct)):
                    file.write(f'\t\tFinal regularized gradients of Theta{layer}:\n')
                    for row in finGradients[layer]:
                        file.write('\t\t')
                        for val in row:
                            file.write(f'{val:9.5f}')
                        file.write('\n')
                    file.write('\n')
                file.write('\n')

        if (not normalized):
            normSamples = self.normalizeSamples(np.array(data, dtype=object))
        else:
            normSamples = data

        if self.demo:
            outputTrainDataInfo(normSamples)

        for epoch in range(numEpochs):
            if numEpochs == 1 and batchSize == 1 and not replace:
                batch = normSamples
            else:
                indices = np.random.choice(len(normSamples["x"]), size=int(batchSize*len(normSamples["x"])), replace=replace)
                batch = {"x": normSamples["x"][indices], "y": normSamples["y"][indices]}
            
            actvns = self.predictVector(batch['x'], True)
            if self.demo:
                outputFeedForwardInfo(batch, actvns)
            
            backPropResultObj = backProp(batch, actvns, self.demo)
            if self.demo:
                outputBackPropInfo(backPropResultObj, batch)

    def predict(self, data):
        result = self.normalizeSamples(data)

        outputVectors = self.predictVector(result["x"])

        outputLabels = []
        for outputVector in outputVectors:
            outputLabel = []
            offset = 0
            for col in self.dataStruct["labelCols"]:
                if self.dataStruct["colsDescr"][col]["type"] == "labelNum":
                    outputLabel.append(outputVector[offset])
                    offset += 1
                if self.dataStruct["colsDescr"][col]["type"] == "labelCat":
                    outputLabel.append(self.dataStruct["colsDescr"][col]["oneHotRev"][np.argmax(outputVector[offset:(offset+self.dataStruct["colsDescr"][col]["oneHotLen"])])])
                    offset += self.dataStruct["colsDescr"][col]["oneHotLen"]
            outputLabels.append(outputLabel)

        return outputLabels
    

#Example of how to run the neural network on any dataset 
#(with any number of numeric and categorical features as well as with any number of labels of numeric and categorical types.
#String categorical values for features and labels are possible):

#data = readCSVtoArr2D('datasets/cmc.data', <RowToBeRemoved>) #<RowToBeRemoved> is optional. It can be used to remove column names.
#dataStruct = NeuralNetwork.getDataStruct(data, ['num','cat','cat','num','cat','cat','cat','cat','cat','labelCat'])
#neuralNetwork = NeuralNetwork(dataStruct)
#neuralNetwork.train(data)
#neuralNetwork.predict(removeLabels(data, dataStruct["labelCols"]))