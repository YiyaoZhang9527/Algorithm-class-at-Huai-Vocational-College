import numpy as np
import math
import random
import matplotlib.pyplot as plt

import platform

system = platform.system()
if system == "Linux":
    plt.rcParams['font.sans-serif'] = ["Noto Sans CJK JP"]
elif system == "Darwin":
    plt.rcParams['font.sans-serif'] = ["Kaiti SC"]
plt.rcParams['axes.unicode_minus'] = False

NUM_OF_DATA = 100

# --------数据生成-------------
def tag_entry(x,y,r=1):
    """
    打标函数
    """
    if x**2+y**2<r:
        tag = 0
    else:
        tag = 1
    return tag

def create_data(num_of_data):
    """
    创建数据
    """
    entry_list = []
    for i in range(num_of_data):
        x = random.uniform(-2,2)
        y = random.uniform(-2,2)
        tag = tag_entry(x,y)
        entry = [x,y,tag]
        entry_list.append(entry)
    return np.array(entry_list)


# —————————— 可视化——————————————
def plot_data(data,title):
    color = []
    for i in data[:,2]:
        if i == 0:
            color.append("orange")
        else:
            color.append("blue")
    plt.figure(figsize=(16,16))
    plt.scatter(data[:,0],data[:,1],c=color)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    data = create_data(NUM_OF_DATA)
    # print(data)
    plot_data(data,title="test")
