import numpy as np
import time
from time import time
def sigmoid(n):
    return 1/(1+exp(-n))

def feedForward(n):
    A = np.ones((n,n));
    B = np.ones((n,n));
    C = np.ones((n,n));
    
    return np.array(map(sigmoid, np.dot(A,B) + C))

l = [16,64,256,2048,8192,16384,32768]
for x in l:
    start_time = time()
    feedForward(x)
    print("Time for", x, " : ", time() - start_time, "s")
