
from classes import *
import time

print("\nThis Neural Network has only one hidden layer!\n")

time.sleep(1)
net = NARNET(filename='btc_usd', numofNeurons=4, numofDelayedInputs=4, hidden_activation='lin', out_activation='lin', numofEpochs=100000)
net.train()
net.plot()
net.predictafterdays()


