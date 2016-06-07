#coding=utf-8
import pybrain
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer,SigmoidLayer
from pybrain.structure import  FullConnection
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer,LinearLayer,SigmoidLayer,SoftmaxLayer
from pybrain.tools.validation import CrossValidator,Validator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# BP Test
# create data x and y
x1=[float(x) for x in range(-100,400)]
x2=[float(x) for x in range(-200,800,2)]
x3=[float(x) for x in range(-500,1000,3)]
num=len(x1);
y=[];
for i in range(num):
    y.append(x1[i]**2+x2[i]**2+x3[i]**2)

    # min-max handle data
x1=[(x-min(x1))/(max(x1)-min(x1)) for x in x1]
x2=[(x-min(x2))/(max(x2)-min(x2)) for x in x2]
x3=[(x-min(x3))/(max(x3)-min(x3)) for x in x3]
y=[(x-min(y))/(max(y)-min(y)) for x in y]

# transform x y be array format
x=np.array([x1,x2,x3]).T
y=np.array(y)
xdim=x.shape[1]  #input dimention
ydim=1  #output dimention

# create supervise dataset
DS=SupervisedDataSet(xdim,ydim);
for i in range(num):
    DS.addSample(x[i],y[i])

train,test=DS.splitWithProportion(0.75)
# DS['input']      value of input x
# DS['target']    value of output y
# DS.clear()     clear data

# create nerve net
ann=buildNetwork(xdim,10,5,ydim,hiddenclass=TanhLayer,outclass=LinearLayer)
# BP train
trainer=BackpropTrainer(ann,dataset=train,learningrate=0.1,momentum=0.1,verbose=True)
# trainer.trainEpochs(epochs=20) #times of training
trainer.trainUntilConvergence(maxEpochs=50) #times of training

# forecast and draw
# forecast test
output=ann.activateOnDataset(test);
# ann.activate(onedata) can only test one
# format output
out=[]
for i in output:
    out.append(i[0])

# draw with pandas
df=pd.DataFrame(out,columns=['predict'])
df['real']=test['target']
df1=df.sort_values(by='real')
df1.index=range(df.shape[0])
df.plot(kind='line')
plt.show()

# # 创建前馈神经网络
# n=FeedForwardNetwork();
# # 创建层
# inLayer=LinearLayer(2);
# hiddenLayer=SigmoidLayer(3);
# outLayer=LinearLayer(1);
# # 网络中加入层
# n.addInputModule(inLayer);
# n.addModule(hiddenLayer);
# n.addOutputModule(outLayer);
# # 创建全联系
# in_to_hidden=FullConnection(inLayer,hiddenLayer);
# hidden_to_out=FullConnection(hiddenLayer,outLayer);
# # 联系加入网络
# n.addConnection(in_to_hidden);
# n.addConnection(hidden_to_out);
# #初始化网络
# n.sortModules();
# #激活层，参数为输入
# print n.activate([1,2])
# # 查看参数
# print in_to_hidden.params
# print hidden_to_out.params
# print n.params
# print n
# print n['LinearLayer-3']
