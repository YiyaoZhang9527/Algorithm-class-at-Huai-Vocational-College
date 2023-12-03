import numpy as np


def F(x):
    return x.sum()


pop = np.random.uniform(0,1,size=(10,10))
# print(pop)

for _ in pop:
    print(0,_)
    f_values = _
    print("f0",f_values)
    f_values = np.reshape(a=f_values,newshape=(10,1))
    print("f1",f_values)
    _ = f_values
    print(1,_)
    
    
    