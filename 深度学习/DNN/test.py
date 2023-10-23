import numpy as np


a01 = 0.9
a02 = 0.5
a03 = 0.7

a11 = -0.8
a12 = -0.5
a13 = 0.6

a21 = -1
a22 = 1
a23 = -0.5


a31 = 0.1
a32 = -0.6
a33 = 0.1


a41 = -0.1
a42 = 0.6
a43 = 0.5

# inputs = np.array(
#     [
#         [a01,a02,a03]
#         ,[a11,a12,a13]
#         ,[a21,a22,a23]
#         ,[a31,a32,a33]
#         ,[a41,a42,a43]
#     ]
# )

inputs = np.array(
    [
        [a01,a02]
        ,[a11,a12]
        ,[a21,a22]
        ,[a31,a32]
        ,[a41,a42]
    ]
)


predicted = np.array(
    [
        [a01,a02]
        ,[a11,a12]
        ,[a21,a22]
        ,[a31,a32]
        ,[a41,a42]
    ]
)

real = np.array([1,0,1,0,1])


def precise_loss_function(predicted,real):
    real_maxtrix = np.zeros((len(real),2))
    real_maxtrix[:,1] = real
    real_maxtrix[:,0] = 1-real
    prodict = np.sum(predicted*real_maxtrix,axis=1)
    return 1-prodict

def get_final_layer_preAct_damands(predicted_value,target_vector):
    """
    需求函数
    """
    target = np.zeros((len(target_vector),2))
    target[:,1] = target_vector
    target[:,0] = 1-target_vector
    for i in range(len(target_vector)):
        if np.dot(target[i],predicted_value[i]) > 0.5:
            target[i] = np.array([0,0])
        else:
            target[i] = (target[i]-0.5)*2
    return target

if __name__ =="__main__":
    print(get_final_layer_preAct_damands(predicted,real))