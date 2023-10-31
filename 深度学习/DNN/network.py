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

def vector_normalize(array):
    """
    向量标准化函数
    """
    abs_number = np.absolute(array)
    max_number = np.max(abs_number)
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
    精确损失函数
    """
    real_maxtrix = np.zeros((len(real),2))
    real_maxtrix[:,1] = real
    real_maxtrix[:,0] = 1-real
    prodict = np.sum(predicted*real_maxtrix,axis=1)
    return 1-prodict

def loss_function(predicted,real):
    condition = (predicted > 0.5)
    binary_predicted = np.where(condition,1,0)

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
    
    def network_backword(self,layer_outputs,target_vector):
        """
        反向传播网络
        """
        # 备用网络
        backup_network = copy.deepcopy(self)
        preAct_demands = get_final_layer_preAct_damands(layer_outputs[-1],target_vector)
        # layers_lenght = len(self.layers)
        for i in range(len(self.layers)):
            ## 倒序取层
            layerNo = len(self.layers)-(1+i)
            layer = backup_network.layers[layerNo]
            if i != 0:
                layer.biases += LEARNING_RATE * np.mean(preAct_demands,axis=0)
                # 再次标准化防止爆炸
                layer.biases = vector_normalize(layer.biases)
            # 更新
            outputs = layer_outputs[len(layer_outputs)-(2+i)]
            results_list = layer.layer_backward(outputs,preAct_demands)
            preAct_demands = results_list[0]
            weights_adjust_matrix  = results_list[1]
            # 更新权重矩阵的反向传播
            layer.weights += LEARNING_RATE * weights_adjust_matrix
            layer.weights = normalize(layer.weights)

        return backup_network
    
    def one_batch_train(self,batch):
        inputs = batch[:,0:2]
        tatgets = copy.deepcopy(batch[:,2]).astype(int)
        outputs = self.network_forward(inputs)
        precise_loss = precise_loss_function(outputs[-1],tatgets)
        loss = loss_function(outputs[-1],tatgets)
        # 如果损失值小于一个数就不训练了
        if np.mean(precise_loss) <= 0.1:
            print("No need for training")
        else:
            backup_network = self.network_backword(outputs,tatgets)
            backup_outputs = backup_network.network_forward(inputs)
            backup_precise_loss = precise_loss_function(backup_outputs[-1], tatgets)
            backup_loss = loss_function(backup_outputs[-1],tatgets)
            ## 如果新网络的损失函数比老网络小，我们就更新网络
            if np.mean(precise_loss) >= np.mean(backup_precise_loss) or  np.mean(loss) >= np.mean(backup_loss):
                # precise_loss = backup_precise_loss
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()
                    self.layers[i].biases = backup_network.layers[i].biases.copy()
                print("Improved 有所改善")
            else:
                print("No Improved 没有改善")

        print("-------------------------------------------------------------")










if __name__ == '__main__':
    BATCH_SIZE = 100
    
    data = cp.create_data(BATCH_SIZE)
    cp.plot_data(data, title="标准答案")
    # x = data[:,0:2]
    # tatgets = copy.deepcopy(data[:,2])
    # print(f"x:\n{x}")
    # print(f"x的维度:{x.shape}")
    n_inputs = 2
    # 因为y只有两个结果，所以输出的结果也是2个
    n_outputs = 2
    # 定义神经网络的层，第一个数要与输入的批次相同
    NETWORK_SHAPE = [n_inputs,3,4,5,n_outputs]
    #批大小
    
    # 学习率
    LEARNING_RATE = 0.1

     #-------------------测试---------------------
    # 建立一个网络
    network = Network(NETWORK_SHAPE)
    print(f"每层网络用了多少个神经元：{network.neurons}，一共用了几层网络：{len(network.neurons)}")
    network.one_batch_train(data)


    # # 建立前馈网络
    # outputs = network.network_forward(x)
    # classification = classify(outputs[-1])
    # print(classification)
    # data[:,2] = classification
    # print(data)
    # cp.plot_data(data,"训练之前的数据")

    # backup_network = network.network_backword(outputs,tatgets)
    # new_outputs = backup_network.network_forward(x)
    # new_classification = classify(new_outputs[-1])
    # data[:,2] = new_classification

    # cp.plot_data(data,"训练之后的数据")





    # classification = classify(outputs[-1])
    # data[:,2] = classification
    # print(f"data:\n{data}")
    # cp.plot_data(data, title="Before traning")

    # loss = precise_loss_function(outputs[-1],tatgets)
    # print(f"loss:\n{loss}")
    # demands = get_final_layer_preAct_damands(outputs[-1],tatgets)
    # print(f'demands:\n{demands}')
    
    # #—----------- 测试调整矩阵----------------
    # adjust_matrix = network.layers[-1].get_weights_adjust_matrix(outputs[-2],demands)
    # print(f"adjust_matrix:\n{adjust_matrix}")
    # # print(precise_loss_function(y,outputs))
    # #-------------测试反向传播----------------
    # layer_backward = network.layers[-1].layer_backward(outputs[-2],demands)
    # print(f"layer_backward:{layer_backward}")


