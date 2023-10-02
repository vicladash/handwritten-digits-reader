import gzip
import pickle
import numpy as np


def loadData():
  f = gzip.open("mnist.pkl.gz", "rb")
  trainData, valData, testData = pickle.load(f, encoding="latin1")
  f.close()
  return (trainData, valData, testData)


def loadDataWrapper():
  trd, vad, ted = loadData()
  trainInputs = [np.reshape(x, (784, 1)) for x in trd[0]]
  trainResults = [vectorizedResult(y) for y in trd[1]]
  trainData = list(zip(trainInputs, trainResults))
  valInputs = [np.reshape(x, (784, 1)) for x in vad[0]]
  valData = list(zip(valInputs, vad[1]))
  testInputs = [np.reshape(x, (784, 1)) for x in ted[0]]
  testData = list(zip(testInputs, ted[1]))
  return (trainData, valData, testData)


def vectorizedResult(j):
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e
