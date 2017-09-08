
from classes import *
import time

print("\nThis Neural Network has only 1 hidden layer!\n")
time.sleep(1)
net = NARNET(filename='btc_usd', numofNeurons=4, numofDelayedInputs=3, hidden_activation='lin', out_activation='lin', numofEpochs=50000)
net.train()
net.predict()


