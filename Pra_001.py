
from tensorflow.examples.tutorials.mnist import input_data   
import tensorflow as tf 
import numpy as np  

#基本参数设置
batchSize = 30   
lr = 0.005       
iter = 1000000   
saveInter = 100  
sample_size = 55000  


def predict(X):   
    num = X.shape[0]  
    result = [] 
    for i in range(num):  
        if X[i]>0.5: 
            result.append(1.0) 
        else: 
            result.append(0.0)  
    return result 

# 加载数据集并解压到./MNIST文件夹下
def loadData(): 
    file = "./MNIST" 
    mnist = input_data.read_data_sets(file, one_hot=True)  
    return mnist 


def create_placeholder(n_x=784,n_y=0): 
    X = tf.placeholder(tf.float32,shape=[None,n_x],name='X')   
    Y = tf.placeholder(tf.float32, shape=[None,], name='Y')  
    return X,Y  

def initialize_parameters(): 
    W = tf.Variable(tf.zeros([784,1]), name="weight")  
    b = tf.Variable(tf.zeros([1,1]), name="bias") 
    parameters={'W': W,  
                'b': b}  
    return parameters 

# 定义网络模型
def forward_propagation(X,parameters):  
    W = parameters['W']  # 参数权重 W
    b = parameters['b']  # 参数偏置 b

    Z1=tf.matmul(X, W) + b  
    A1=tf.nn.sigmoid(Z1, name='sigmoid_forward')  
    A1 = tf.clip_by_value(A1, 0.001, 1.0, name="clip_forward")  
    return A1 

def compute_cost(y_,y,W):  
    cross_entropy = -(1.0/batchSize)*tf.reduce_sum((1.0-y_)*tf.log(1.0-y)) 
    return cross_entropy   

# 模型搭建、训练、存储
def model(mnist,Num): 
    x,y_ = create_placeholder(784, 0) 
    parameters = initialize_parameters() 
    A1 = forward_propagation(x, parameters)   

    global_step = tf.Variable(0)  
    learning_rate = tf.train.exponential_decay(lr,global_step,decay_steps=sample_size/batchSize,decay_rate=0.98,staircase=True) # 设置指数衰减的 学习率

    cost = compute_cost(y_, A1,parameters['W']) 
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step) 
    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer()) 

    #利用全部样本对模型进行测试
    testbatchX = mnist.train.images  # 导入 mnist 数据中的训练集 图片
    testbatchY = mnist.train.labels  # 导入 mnist 数据中的训练集 标签

    modelLast = [] 
    logName = "./log"+str(Num)+".txt" 

    saver = tf.train.Saver(max_to_keep=4)  
    pf = open(logName, "w") 
    for i in range(iter): 
        batch = mnist.train.next_batch(batchSize) 
        batchX = batch[0] 
        batchY = batch[1] 
        #执行训练
        train_step.run(feed_dict={x: batchX, y_: batchY[:,Num]})  

        if i % saveInter == 0:  
            [total_cross_entropy,pred,Wsum,lrr] = sess.run([cost,A1,parameters['W'],learning_rate],feed_dict={x:batchX,y_:batchY[:,Num]}) # 调用 sess.run， 启动 tensoflow
            pred1 = predict(pred) 

            #保存当前模型的学习率lr、在minibatch上的测试精度
            print('lr:{:f},train Set Accuracy: {:f}'.format(lrr,(np.mean(pred1 == batchY[:,Num]) * 100))) # 输出训练集的准确率等
            pf.write('lr:{:f},train Set Accuracy: {:f}\n'.format(lrr,(np.mean(pred1 == batchY[:,Num]) * 100))) # 写入训练集的准确率

            #保存迭代次数、cross entropy
            print("handwrite: %d, iterate times: %d , cross entropy:%g"%(Num,i,total_cross_entropy)) # 输出迭代次数，交叉熵损失函数等
            pf.write("handwrite: %d, iterate times: %d , cross entropy:%g, W sum is: %g\n" %(Num,i,total_cross_entropy,np.sum(Wsum))) # 写入出迭代次数，交叉熵损失函数等

            #保存当前参数状态、测试testbatch上的精度
            [testpred] = sess.run([A1],feed_dict={x: testbatchX, y_: testbatchY[:, Num]})  # 调用 sess.run， 启动 tensoflow
            testpred1 = predict(testpred)   
            print('predict sum is: {:f},Testing Set Accuracy: {:f}\n'.format(np.sum(testpred1),(np.mean(testpred1 == testbatchY[:, Num]) * 100)))  # 输出测试集的准确率等
            pf.write('predict sum is: {:f},Testing Set Accuracy: {:f}\n'.format(np.sum(testpred1),(np.mean(testpred1 == testbatchY[:,Num]) * 100))) # 写入测试集的准确率等
            pf.write("\n")

            #保存当前模型
            saveName = "model/my-model-" + str(Num) 
            saver.save(sess, saveName, global_step=i)
            pf.write("save model completed\n") 

            if total_cross_entropy != total_cross_entropy: 
                print("is nan, stop")
                pf.write("is nan, stop\n") 
                modelLast = "model/my-model-" + str(Num)+str(i-saveInter) # 模型文件名为  "model/my-model-" + str(Num)+str(i-saveInter)
                break; # break 跳出循环
    pf.close() # close 关闭打开的文件
    return modelLast # 返回 modelLast

# 模型测试
def test_model(): 
    mnist = loadData()
    classNum = 10 
    modelNames = [] 
    logName = "./logModelNames.txt" 
    pf = open(logName, "w") 

    for i in range(classNum):
        modelNames.append(model(mnist,i)) 
        pf.write(modelNames[i]) 
        pf.write("\n") 
    pf.close() 

if __name__ == '__main__': 
    test_model() 

