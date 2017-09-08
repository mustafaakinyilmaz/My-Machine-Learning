import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# one hidden layer
numofNeurons = 10
numofDelayedInputs = 5


filename = 'btc_usd'
file = '/home/akinyilmaz/Desktop/Machine/forecast/' + filename + '.csv'

df = pd.read_csv(file)
df = df.dropna(axis=0, how='any')
price = df['Price']


X1 = price[:-5]
print(len(X1))
print(len(price))
X2 = price[1:-4]
X3 = price[2:-3]
X4 = price[3:-2]
X5 = price[4:-1]
X = np.vstack([X1, X2, X3, X4, X5])
Y1 = price[5:]
Y = np.vstack([Y1])


D = numofDelayedInputs
M = numofNeurons

W1 = np.random.randn(D,M)
W2 = np.random.randn(M,1)

def sigmoid(inp, deriv= False):
    fx = 1/(1+np.exp(-inp))
    if deriv == True:
        return fx*(1-fx)
    else:
        return fx

def tanh(inp, deriv= False):
    fx = (2 / (1 + np.exp(-2 * inp))) - 1
    if deriv == True:
        return 1-np.square(fx)
    else:
        return fx

def lin(inp, deriv= False):
    fx = inp
    if deriv == True:
        return 1
    else:
        return fx

def relu(inp, deriv= False):
    fx = inp*(inp>0)
    if deriv == True:
        return inp > 0
    else:
        return fx


syn_in = W1
syn_ou = W2
numofEpochs = 60000

for j in range(numofEpochs):
    # layers
    l0 = X
    l1 = relu(np.dot(l0.T,syn_in))
    l2 = lin(np.dot(l1,syn_ou))
    #l2_error = np.sqrt(np.square(Y - l2.T))
    l2_error = Y.T - l2
    l2_delta = l2_error * lin(l2, deriv= True)
    l1_error = np.dot(l2_delta,syn_ou.T)
    l1_delta = l1_error * relu(l1, deriv= True)
    if j % 10000 == 0:
        print("error: %s"%(str(np.mean(np.abs(l2_error)))))
        print("---")

    rate = 0.00000000001
    syn_ou += rate*np.dot(l1.T,l2_delta)
    syn_in += rate*np.dot(l0,l1_delta)

"""print(syn_ou)

inp = np.vstack([price[-6:-1]])
print(inp)
a1 = relu(np.dot(inp,syn_in))
a2 = relu(np.dot(a1,syn_ou))
print(a2)
print(price[len(price)-1])"""

l0 = X
l1 = relu(np.dot(l0.T, syn_in))
l2 = lin(np.dot(l1, syn_ou))

diff = l2 - Y.T
print(max(abs(diff)))
print(min(abs(diff)))
days = np.arange(numofDelayedInputs,len(price),1)
#plt.plot(days, Y.T, 'b--', days, l2, 'r--')
plt.plot(days, diff)

plt.show()

"""inp = np.vstack([price[-5:]])
a1 = relu(np.dot(inp,syn_in))
a2 = relu(np.dot(a1,syn_ou))
print(a2)"""












