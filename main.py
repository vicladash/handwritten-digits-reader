import MNISTloader
import net

trainData, valData, testData = MNISTloader.loadDataWrapper()
network = net.Net([784, 30, 10])
network.SGD(trainData, 30, 10, 3.0, testData=testData)
