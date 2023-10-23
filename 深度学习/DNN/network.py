import numpy as np
import creat_data_and_plot as cp
import numpy as np
import copy


def activation_Relu(inputs):
    """
    Relu激活函数
    """
    return np.maximum(0,inputs) 

def activation_softmax(inputs):
    """
    softmax 激活函数
    """
    max_values = np.max(inputs,axis=1,keepdims=True)
    slided_input = inputs-max_values
    exp_value = np.exp(slided_input)
    norm_base = np.sum(exp_value,axis=1,keepdims=True)
    norm_values = exp_value/norm_base 
    return norm_values
        
def create_weights(n_inputs,n_neurons):
    """
    创建权重
    """
    return np.random.randn(n_inputs,n_neurons)

def create_biases(n_neurons):
    """
    创建偏置
    """
    return np.random.randn(n_neurons)

def normalize(array):
    """
    标准化函数让值永远处于-1，1之间
    """
    abs_number = np.absolute(array)
    max_number = np.max(abs_number,axis=1,keepdims=True)
    scale_rate = np.where(max_number==0,0,1/max_number)
    norm = array * scale_rate
    return norm

def classify(probability):
    """
    分类函数
    """
    clssification = probability[:,1]
    return np.rint(clssification)


def precise_loss_function(predicted,real):
    """
    损失函数
    """
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

class Layer:
    def __init__(self,n_inputs,n_neurons):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs 
        self.weights = np.random.randn(self.n_inputs,self.n_neurons)
        self.biases = np.random.randn(self.n_neurons)
    
    def layer_forward(self,inputs):
        sum_ = np.dot(inputs,self.weights)+self.biases
#         self.output = activation_Relu(sum_)
#         self.next_n_input = self.output.shape[-1]
#         return self.output
        return sum_
    
    def layer_backward(self,preWeight_Values,afterWeights_demands):
        """
        层反向传播
        afterWeights_demand ：
        preWeight_Values ：权重矩阵相乘之前的output

        输出：
        norm_preActs_demands 前一条的需求之
        norm_weights_adjust_matrix ：前一条的调整矩阵

        """
        preWeight_demands = np.dot(afterWeights_demands,self.weights.T)
        # Rule 激活函数的导数
        comdition = preWeight_Values > 0 
        value_derivatives = np.where(comdition,1,0)
        # 激活函数之前的需求值
        preActs_demands = value_derivatives*preWeight_demands
        # 标准化防止梯度发散
        norm_preActs_demands = normalize(preActs_demands)
        # 计算调整矩阵
        weights_adjust_matrix = self.get_weights_adjust_matrix(preWeight_Values,afterWeights_demands)
        # 再次标准化
        norm_weights_adjust_matrix = normalize(weights_adjust_matrix)

        return (norm_preActs_demands,norm_weights_adjust_matrix)



    
    def get_weights_adjust_matrix(self,preWeights_values,aftWeights_demands):
        """
        调整矩阵
        """
        plain_weights = np.ones(self.weights.shape)
        weights_adjust_matrix = np.zeros(self.weights.shape)
        plain_weights_T = plain_weights.T

        for i in range(BATCH_SIZE):
            weights_adjust_matrix += (plain_weights_T*preWeights_values[i,:]).T*aftWeights_demands[i,:]
        weights_adjust_matrix = weights_adjust_matrix/BATCH_SIZE
        return weights_adjust_matrix




class Network:
    def __init__(self,network_shape:list):
        self.shape = network_shape
        self.layers = []
        self.neurons = []
        for i in range(len(network_shape)-1):
            # 输入维度m,n的n值
            n_inputs = network_shape[i]
            # 神经元数量
            n_neurons = network_shape[i+1]
            layer = Layer(n_inputs,n_neurons)
            self.layers.append(layer)
            self.neurons.append(n_neurons)
    
    ## 前馈网络
    def network_forward(self,inputs):
        """
        前馈网络
        """
        outputs = [inputs]
        layer_lenght = len(self.layers) 
        for i in range(layer_lenght):
            layer_sum = self.layers[i].layer_forward(outputs[i])
            if i < layer_lenght-1:
                # 前期用Rule函数激活
                layer_output = activation_Relu(layer_sum)
                # 对输出值标准化，让他永远处于-1，1之间防止爆炸
                layer_output = normalize(layer_output)
                
            else:
                ## 最后一行用softmax函数激活
                layer_output = activation_softmax(layer_sum)
                # print(f"softmax_output:\n{layer_output}")
            outputs.append(layer_output)
        return outputs
            



if __name__ == '__main__':
    m = 10
    data = cp.create_data(m)
    cp.plot_data(data, title="Right classification")
    x = data[:,0:2]
    tatgets = copy.deepcopy(data[:,2])
    # print(f"x:\n{x}")
    print(f"x的维度:{x.shape}")
    n_inputs = x.shape[-1]
    # 因为y只有两个结果，所以输出的结果也是2个
    n_outputs = 2
    # 定义神经网络的层，第一个数要与输入的批次相同
    NETWORK_SHAPE = [n_inputs,3,4,5,n_outputs]
    #批大小
    BATCH_SIZE = m

    # 建立一个网络
    network = Network(NETWORK_SHAPE)
    print(f"每层网络用了多少个神经元：{network.neurons}，一共用了几层网络：{len(network.neurons)}")
    # 建立前馈网络
    outputs = network.network_forward(x)
    classification = classify(outputs[-1])
    data[:,2] = classification
    print(f"data:\n{data}")
    cp.plot_data(data, title="Before traning")

    loss = precise_loss_function(outputs[-1],tatgets)
    print(f"loss:\n{loss}")
    demands = get_final_layer_preAct_damands(outputs[-1],tatgets)
    print(f'demands:\n{demands}')
    
    #—----------- 测试调整矩阵----------------
    adjust_matrix = network.layers[-1].get_weights_adjust_matrix(outputs[-2],demands)
    print(f"adjust_matrix:\n{adjust_matrix}")
    # print(precise_loss_function(y,outputs))
    #-------------测试反向传播----------------
    layer_backward = network.layers[-1].layer_backward(outputs[-2],demands)
    print(f"layer_backward:{layer_backward}")