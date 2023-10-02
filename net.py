import random
import numpy as np


class Net:

  def __init__(self, sizes):
    self.sizes = sizes
    self.layerNum = len(sizes)
    self.weights = [
      np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
    ]
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

  def SGD(self, trainData, epochNum, miniBatchSize, eta, testData=None):
    if testData: testNum = len(testData)
    trainNum = len(trainData)
    for j in range(epochNum):
      random.shuffle(trainData)
      miniBatches = [
        trainData[k:k + miniBatchSize]
        for k in range(0, trainNum, miniBatchSize)
      ]
      for miniBatch in miniBatches:
        self.updateMiniBatch(miniBatch, eta)
      if testData:
        print("Epoch " + str(j) + ": " + str(self.evaluate(testData)) + " / " +
              str(testNum))
      else:
        print("Epoch " + str(j) + " complete")

  def updateMiniBatch(self, miniBatch, eta):

    #nablaW = [np.zeros(w.shape) for w in self.weights]
    #nablaB = [np.zeros(b.shape) for b in self.biases]
    #for x, y in miniBatch:
    #  deltaNablaW, deltaNablaB = self.backprop(x, y)
    #  nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)]
    #  nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)]
    nablaW, nablaB = self.backpropSim(miniBatch)

    self.weights = [
      w - eta / len(miniBatch) * nw for w, nw in zip(self.weights, nablaW)
    ]
    self.biases = [
      b - eta / len(miniBatch) * nb for b, nb in zip(self.biases, nablaB)
    ]

  def evaluate(self, testData):
    testResults = [(np.argmax(self.feedforward(x)), y) for x, y in testData]
    return sum(int(x == y) for x, y in testResults)

  def backprop(self, x, y):
    nablaW = [np.zeros(w.shape) for w in self.weights]
    nablaB = [np.zeros(b.shape) for b in self.biases]
    activation = x
    activations = [x]
    zs = []
    for w, b in zip(self.weights, self.biases):
      z = np.dot(w, activation) + b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    delta = self.costDerivative(activations[-1], y) * sigmoidPrime(zs[-1])
    nablaW[-1] = np.dot(delta, activations[-2].transpose())
    nablaB[-1] = delta
    for l in range(2, self.layerNum):
      z = zs[-l]
      sp = sigmoidPrime(z)
      delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
      nablaW[-l] = np.dot(delta, activations[-l - 1].transpose())
      nablaB[-l] = delta
    return (nablaW, nablaB)

  def backpropSim(self, miniBatch):
    nablaW = [np.zeros(w.shape) for w in self.weights]
    nablaB = [np.zeros(b.shape) for b in self.biases]
    activation = np.concatenate([x for x, y in miniBatch], axis=1)
    activations = [activation]
    zs = []
    y = np.concatenate([y for x, y in miniBatch], axis=1)
    for w, b in zip(self.weights, self.biases):
      z = np.dot(w, activation) + b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    delta = self.costDerivative(activations[-1], y) * sigmoidPrime(zs[-1])
    nablaW[-1] = np.dot(delta, activations[-2].transpose())
    nablaB[-1] = np.sum(delta, axis=1, keepdims=True)
    for l in range(2, self.layerNum):
      z = zs[-l]
      sp = sigmoidPrime(z)
      delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
      nablaW[-l] = np.dot(delta, activations[-l - 1].transpose())
      nablaB[-l] = np.sum(delta, axis=1, keepdims=True)
    return (nablaW, nablaB)

  def feedforward(self, a):
    for w, b in zip(self.weights, self.biases):
      a = sigmoid(np.dot(w, a) + b)
    return a

  def costDerivative(self, outputActivations, y):
    return outputActivations - y


def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))


def sigmoidPrime(z):
  return sigmoid(z) * (1.0 - sigmoid(z))
