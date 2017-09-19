import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ACTIVATION FUNCTIONS
def sigmoid(inp, deriv=False):
    fx = 1 / (1 + np.exp(-inp))
    if deriv == True:
        return fx * (1 - fx)
    else:
        return fx
def tanh(inp, deriv=False):
    fx = (2 / (1 + np.exp(-2 * inp))) - 1
    if deriv == True:
        return 1 - np.square(fx)
    else:
        return fx
def lin(inp, deriv=False):
    fx = inp
    if deriv == True:
        return 1
    else:
        return fx
def relu(inp, deriv=False):
    fx = inp * (inp > 0)
    if deriv == True:
        return inp > 0
    else:
        return fx


class NARNET:
    def __init__(self, filename, numofNeurons, numofDelayedInputs, hidden_activation, out_activation, numofEpochs):
        self.filename = filename
        self.numofNeurons = numofNeurons
        self.numofDelayedInputs = numofDelayedInputs
        self.numofEpochs = numofEpochs

        if hidden_activation == 'relu':
            self.hidden_activation = relu
        if hidden_activation == 'lin':
            self.hidden_activation = lin
        if hidden_activation == 'tanh':
            self.hidden_activation = tanh
        if hidden_activation == 'sigmoid':
            self.hidden_activation = sigmoid

        if out_activation == 'relu':
            self.out_activation = relu
        if out_activation == 'lin':
            self.out_activation = lin
        if out_activation == 'tanh':
            self.out_activation = tanh
        if out_activation == 'sigmoid':
            self.out_activation = sigmoid

        self.file = '/home/akinyilmaz/Desktop/Machine/btc_dollar_forecast/' + filename + '.csv'
        self.D = numofDelayedInputs
        self.M = numofNeurons
        self.W1 = np.random.randn(self.D, self.M)
        self.W2 = np.random.randn(self.M, 1)

    def getData(self):
        file = self.file
        df = pd.read_csv(file)
        df = df.dropna(axis=0, how='any')
        price = df['Price']
        date = df['Date']
        return price,date

    def createMatrices(self):
        price = self.getData()[0]
        X_ = []
        for i in range(self.numofDelayedInputs):
            X_.append(list(price[i:i-self.numofDelayedInputs]))

        X = np.array(X_)
        Y1 = price[self.numofDelayedInputs:]
        Y = np.vstack([Y1])
        return X,Y

    """def train(self):
        X = self.createMatrices()[0]
        Y = self.createMatrices()[1]
        print("Training Neural Network...")
        print("------")
        learning_rate = 1e-11
        for j in range(self.numofEpochs):
            #layers
            l_inp = X
            l_hidden = self.hidden_activation(np.dot(l_inp.T,self.W1))
            l_out = self.out_activation(np.dot(l_hidden,self.W2))
            l_out_error = Y.T - l_out
            l_out_delta = l_out_error * self.out_activation(l_out, deriv=True)
            l_hidden_error = np.dot(l_out_delta, self.W2.T)
            l_hidden_delta = l_hidden_error * self.hidden_activation(l_hidden, deriv=True)
            if j % 10000 == 0:
                #print("error: %s" % (str(np.mean(np.abs(l_out_error)))))
                print("cost: %s" % (np.sqrt(sum(np.square(l_out_error)))/np.size(Y)))
                print("-------")
            if j == self.numofEpochs - 1:
                print("cost: %s" % (np.sqrt(sum(np.square(l_out_error)))/np.size(Y)))
                print("-------")


            self.W2 += learning_rate * np.dot(l_hidden.T, l_out_delta)
            self.W1 += learning_rate * np.dot(l_inp, l_hidden_delta)"""

    def train(self):
        X = self.createMatrices()[0]
        Y = self.createMatrices()[1]
        print("Training Neural Network...")
        print("------")
        learning_rate = 1e-11
        costlist = np.array([],dtype=float)
        count = 0

        while True:

            # layers
            l_inp = X
            l_hidden = self.hidden_activation(np.dot(l_inp.T, self.W1))
            l_out = self.out_activation(np.dot(l_hidden, self.W2))
            l_out_error = Y.T - l_out
            l_out_delta = l_out_error * self.out_activation(l_out, deriv=True)
            l_hidden_error = np.dot(l_out_delta, self.W2.T)
            l_hidden_delta = l_hidden_error * self.hidden_activation(l_hidden, deriv=True)

            cost = (np.sqrt(sum(np.square(l_out_error))) / np.size(Y))
            costlist = np.append(costlist,cost)

            if count == 0:
                print("cost: %s"%(cost))
            if count >= 1:

                if (costlist[0] - costlist[1]) / costlist[0] <= 0.0000001:
                    print("cost: %s"%(cost))
                    print("% error: "%((costlist[0] - costlist[1]) / costlist[0]))
                    break
                if count % 1000 == 0:
                    print("cost: %s"%(cost))

                costlist = costlist[1]
            self.W2 += learning_rate * np.dot(l_hidden.T, l_out_delta)
            self.W1 += learning_rate * np.dot(l_inp, l_hidden_delta)
            count += 1
        print(count)


    def plot(self):

        Y = self.createMatrices()[1].T
        print("Plotting predicted vs real")
        l_inp = self.createMatrices()[0]
        l_hidden = self.hidden_activation(np.dot(l_inp.T, self.W1))
        l_out = self.out_activation(np.dot(l_hidden, self.W2))
        days = np.arange(self.numofDelayedInputs, len(self.getData()[0]), 1)

        diff = abs(l_out - Y)

        print("diff mean is %s" % np.mean(diff))

        plt.title("red -> predicted, blue -> real")
        plt.plot(days, Y, 'b', days, l_out, 'r', marker= 'o')
        plt.show()


    def predictafterdays(self):
        days = int(input("Enter how many days:"))
        l_inp = self.createMatrices()[1][0, -self.numofDelayedInputs:]

        for i in range(days):
            l_hidden = self.hidden_activation(np.dot(l_inp.T, self.W1))
            l_out = self.out_activation(np.dot(l_hidden, self.W2))
            l_inp_insert = np.insert(l_inp, self.numofDelayedInputs, l_out, axis=0)
            l_delete = np.delete(l_inp_insert, 0, axis=0)
            l_inp = l_delete

        print("After %d days 1 btc is predicted as %s $" %(days,l_out))
