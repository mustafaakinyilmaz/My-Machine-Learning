import numpy as np
import os


"""a = [1,2,3,4,5,6,7]
array = array([])
print(array)
for i in range(3):
    array = vstack([array, a[i:i-3]])

b = [1,2,3,4,5]

arr = []
for i in range(3):
    arr.append(b[i:i-3])

print(arr)
n = np.array(arr)
print(n.T)
"""


def find_loc(name):
    for root,dir,files in os.walk("/home/akinyilmaz/"):
        if name in files:
            print(root,name)
            break

name = "btc_usd.csv"
find_loc(name)