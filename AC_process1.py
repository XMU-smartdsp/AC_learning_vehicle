#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:16:31 2019

@author: happywjt
"""

#GPU setting
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
import layer as l
import unit 
import numpy as np
import tensorflow as tf
from random import shuffle
from copy import deepcopy as dc
from PIL import Image
import matplotlib.pylab as plt
import scipy.io as sio
from sklearn.metrics import confusion_matrix,classification_report

#function to plot confusion matrix
def plotconfusion(cm,title,num_classes,cmap = plt.cm.binary):
    plt.figure()
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(num_classes))
    plt.xticks(xlocations,xlocations,rotation=90)
    plt.yticks(xlocations,xlocations)
    plt.ylabel('True label')
    plt.xlabel('predicted label')
    savename=title+'.png'
    plt.savefig(savename, format='png')
#import matplotlib.pyplot as plt

#parameter setting
MODEL_INIT = np.load('./bvlc_alexnet_new.npy').item()                                             #Loading pre-trained imagenet model, a pre-downloaded model file are required
CLASS_NUM = 20                                                                                    #Amount of classes to be classify
BATCH_SIZE = 20                                                                                   #Batch size
EPOCH = 50                                                                                        #Epoch of initialization stage
EPOCH_all = 30																					  #进入主动学习后，每次迭代训练的epoch个数
trainset_num = 3                                                                                  #Sample amount of 
ACTIVE_TIME = 5                                                                                   #Query round for active learning
QUEUE_SIZE = 20                                                                                   #High entropy sample amount in every query round
GOOD_SIZE = 20                                                                                    #High conficence sample amount in every query round
TRAIN_TIME = 300
TEST = 5
'''Car Fine grained'''
MODEL_PATH = './model/'                                                 #model saved path
FILE_PATH = './cars_list/'
Experiment_NAME = 'comps_sv_20(3)_2'                                                              #Experiments name, your model saved files will be named under this name
#Experiment_NAME_m = 'comps_sv_20(3)_3'
PRETRAIN_MODEL_NAME = MODEL_PATH + Experiment_NAME +'.ckpt'

#Data selection, source directory, target directory and selected car serial number
data_dir = '/home/happywjt/carpictures/image/'                                                    #Data directory, the format of data storage is in readme file
item_id = [1,15,20,35,48,60,73,86,111,124,137,162,175,188,212,225,238,250,263,276]                #This is the serial number of 20 types of cars selected randomly
#item_id = range(1,101)                                                                           
item_label = [0,4,8,15,16,17,18,19,1,2,3,5,6,7,9,10,11,12,13,14]
#item_label = range(CLASS_NUM)


global active_list_name
global good_list_name


assert(len(item_label)==len(item_id) and len(item_label)==CLASS_NUM)

##############################################################################
##############################################################################
#Model defination
def SharePart(input, drop_out_rate):
    
    def pre_process(input):
        rgb_scaled = input
        Mean = [103.939,116.779,123.68]
        
        red,green,blue = tf.split(rgb_scaled,3,3)
        bgr = tf.concat([
                red - Mean[2],
                green - Mean[1],
                blue - Mean[0]],3)
        return bgr
    
    input = pre_process(input)
    
    with tf.variable_scope('Share_Part'):
        
        conv1 = l.conv2d('conv1',input,(11,11),96,strides = [1,4,4,1],decay = (0.0,0.0),pad='VALID',Init = MODEL_INIT['conv1'])
        maxpool1 = l.max_pooling('maxpool',conv1,3,2)
        norm1 = tf.nn.lrn(maxpool1,depth_radius=2,alpha=2e-05,beta=0.75,name='conv1')
    
        conv2 = l.conv2d_with_group('conv2',norm1,(5,5),256,2,decay = (0.0,0.0),pad = 'SAME', Init = MODEL_INIT['conv2'])
        maxpool2 = l.max_pooling('maxpool2',conv2,3,2)
        norm2 = tf.nn.lrn(maxpool2,depth_radius=2,alpha=2e-05,beta=0.75,name='conv2')

        conv3 = l.conv2d('conv3',norm2,(3,3),384,pad = 'SAME',Init = MODEL_INIT['conv3'])
    
    
        conv4 = l.conv2d_with_group('conv4',conv3,(3,3),384,2,pad = 'SAME',Init = MODEL_INIT['conv4'])
       
        conv5 = l.conv2d_with_group('conv5',conv4,(3,3),256,2,pad = 'SAME',Init = MODEL_INIT['conv5'])
        maxpool5 = l.max_pooling('maxpool5',conv5,3,2)
        print maxpool5.shape
    
        dim=1
        shape = maxpool5.get_shape().as_list()
        for d in shape[1:]:
            dim*=d
    
        reshape = tf.reshape(maxpool5,[-1,dim])
    
        fc6 = l.fully_connect('fc6',reshape,4096,Init = MODEL_INIT['fc6'])
        fc6 = l.dropout('drop_6',fc6,drop_out_rate)
        fc7 = l.fully_connect('fc7',fc6,4096,Init = MODEL_INIT['fc7'])
        fc7 = l.dropout('drop_7',fc7,drop_out_rate)
        
    return fc7

#Mission layer
def MissionPart(input):
    
    with tf.variable_scope('Classifier'):
        result = l.fully_connect('classifier',input,CLASS_NUM,active=None)

    return result

#loss function
def SoftmaxWithLoss(logistic,label):
    
    label = tf.one_hot(label,depth = CLASS_NUM)
    loss = tf.losses.softmax_cross_entropy(label,logistic)
    
    return loss

#training
def train_net(loss,base_lr=0.00001):
    
    
    var_list = tf.trainable_variables()
    trn_list = []
    for i in var_list:
        if 'conv1' not in i.name and 'conv2' not in i.name:
            trn_list.append(i)
            tf.summary.histogram('weight',i)
    
    loss = tf.add_n(tf.get_collection('losses'),name='all_loss')
    opt = tf.train.AdamOptimizer(base_lr).minimize(loss,var_list=trn_list)
    return opt

#test
def Test(logistic,label):
    
    result = tf.cast(tf.argmax(logistic,axis = 1),tf.uint8)
    compare = tf.cast(tf.equal(result,label),tf.float32)
    acc = tf.reduce_mean(compare)
    return acc

#############################################################################################################################
################################################################################################################################
#
#Step1: data processing. Generate training data list, test data list and unlabeled data list. The names of these files are related to Experiment_NAME.
def data_process():
    
    train_list,oracle_samples_list,test_samples_list = unit.GetListTvT(item_id,item_label,data_dir,trainset_num)  
    
    file_train = open(FILE_PATH + Experiment_NAME + '_x_train.txt','w')
    for fp in train_list:
        file_train.write(str(fp))
        file_train.write('\n')
    file_train.close()
    
    file_oracle = open(FILE_PATH + Experiment_NAME + '_x_oracle.txt','w')
    for fp in oracle_samples_list:
        file_oracle.write(str(fp))
        file_oracle.write('\n')
    file_oracle.close()
    
    file_test = open(FILE_PATH + Experiment_NAME + '_x_test.txt','w')
    for fp in test_samples_list:
        file_test.write(str(fp))
        file_test.write('\n')
    file_test.close()
#load data list
def load_process():
    file_train = FILE_PATH + Experiment_NAME + '_x_train.txt'
    file_oracle = FILE_PATH + Experiment_NAME + '_x_oracle.txt'
    file_test = FILE_PATH + Experiment_NAME + '_x_test.txt'
    TrainList = unit.LoadCarTxT(file_train)
    OracleList = unit.LoadCarTxT(file_oracle)
    TestList = unit.LoadCarTxT(file_test)
    TrainData, TrainLabels = unit.Getlist(TrainList)
    OracleData, OracleLabels = unit.Getlist(OracleList)
    TestData, TestLabels = unit.Getlist(TestList)
    UnlabelData = np.concatenate([OracleData, TestData])
    UndataLabels = np.concatenate([OracleLabels, TestLabels])

#Data of initialization and unlabeled data should be merged when training under full supervised manner and this will get the best model
#    TrainData = np.concatenate([TrainData, OracleData])            
#    TrainLabels = np.concatenate([TrainLabels, OracleLabels])
    
    lenn_s = len(TrainData)/BATCH_SIZE
    lenn_t = len(TestData)/BATCH_SIZE
    lenn_u = len(UnlabelData)/BATCH_SIZE
    if len(TrainData)%BATCH_SIZE != 0:
        lenn_s += 1
        TrainData = np.concatenate((TrainData,TrainData[0:(lenn_s*BATCH_SIZE-len(TrainData))]))
        TrainLabels = np.concatenate((TrainLabels,TrainLabels[0:(lenn_s*BATCH_SIZE-len(TrainLabels))]))
        
    if len(TestData)%BATCH_SIZE != 0:
        lenn_t += 1
        TestData = np.concatenate((TestData,TestData[0:(lenn_t*BATCH_SIZE-len(TestData))]))
        TestLabels = np.concatenate((TestLabels,TestLabels[0:(lenn_t*BATCH_SIZE-len(TestLabels))]))
    
    if len(UnlabelData)%BATCH_SIZE != 0:
        lenn_u += 1
        UnlabelData = np.concatenate((UnlabelData,UnlabelData[0:(lenn_u*BATCH_SIZE-len(UnlabelData))]))
        UndataLabels = np.concatenate((UndataLabels,TestLabels[0:(lenn_u*BATCH_SIZE-len(UndataLabels))]))
        
    return TrainData, TrainLabels, OracleData,  OracleLabels, TestData, TestLabels, UnlabelData, UndataLabels, lenn_s, lenn_t, lenn_u
	
######################################################################################
######################################################################################
#Model initialization: using initial dataset to pretrain network, and keep the best model in validation dataset. Model name is related to Experiment_NAME	
def pretrain():
    #load data
    TrainData, TrainLabels,_ ,_ ,\
    _, _, UnlabelData, UndataLabels,\
    lenn_s, _, lenn_u = load_process()
    
    batch = tf.placeholder(tf.float32,[None,unit.H,unit.W,unit.Channel])
    label = tf.placeholder(tf.uint8,[None])
    keep_prop = tf.placeholder(tf.float32)
    
    feature = SharePart(batch,keep_prop)
    result = MissionPart(feature)
    loss = SoftmaxWithLoss(result,label)
    acc = Test(result,label)
    opt = train_net(loss)
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = False
    init = tf.global_variables_initializer()
    sess = tf.Session(config = config)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)
    

    best_test_acc = 0
    
    train_queue = np.arange(len(TrainData))        #the id of training data
    test_queue = np.arange(len(UnlabelData))       #the id of test data
	#choose data through its id
    for i in range(EPOCH):
        shuffle(train_queue), shuffle(test_queue)
        train_accuracy = 0
        test_accuracy = 0
        test_cost = 0
        for j in range(lenn_s):
            immmg = j
            trainbatch = TrainData[train_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
            trainlabels = TrainLabels[train_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
            sess.run(opt,feed_dict={batch:trainbatch,label:trainlabels,keep_prop:0.5})
            train_accuracy += sess.run(acc,feed_dict={batch:trainbatch,label:trainlabels,keep_prop:1.0})
#        for j in range(len(target_samples_list)/BATCH_SIZE + 1):
        for j in range(lenn_u):
            immmg = j
#            targetbatch,targetlabels = unit.GetBatch(target_samples_list,BATCH_SIZE,j*BATCH_SIZE)
            testbatch = UnlabelData[test_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
            testlabels = UndataLabels[test_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
            test_accuracy+=sess.run(acc ,feed_dict={batch:testbatch,label:testlabels,keep_prop:1.0})
            test_cost+=sess.run(loss,feed_dict={batch:testbatch,label:testlabels,keep_prop:1.0})
        rs = sess.run(merged)
        writer.add_summary(rs, i)
        train_accuracy /= lenn_s
        test_accuracy /= lenn_u
        test_cost /= lenn_u
        
        print "this is the ",i," epoch"
        print "target accuracy is: ",test_accuracy
        print "source accuracy is: ",train_accuracy
        print "target cost is: ",test_cost
        
        if test_accuracy>best_test_acc:
            best_test_acc = test_accuracy
            saver.save(sess,MODEL_PATH+PRETRAIN_MODEL_NAME)
            print "the best test acc is:",best_test_acc
        else:
            print "the best test acc is:",best_test_acc        
     
    return

######################################################################################################################################
####################################################################################################################################
#active learning with random selection strategy
def random_active():
    #load data
    TrainData, TrainLabels, OracleData,  OracleLabels,\
    TestData, TestLabels, _, _, lenn_s, lenn_t, _ = load_process()
    
    batch = tf.placeholder(tf.float32,[None,unit.H,unit.W,unit.Channel])
    label = tf.placeholder(tf.uint8,[None])
    keep_prop = tf.placeholder(tf.float32)
    
    feature = SharePart(batch,keep_prop)
    result = MissionPart(feature)
    loss = SoftmaxWithLoss(result,label)
    acc = Test(result,label)
    opt = train_net(loss)
    
    saver = tf.train.Saver(max_to_keep = ACTIVE_TIME)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    config.gpu_options.allow_growth = False
    init = tf.global_variables_initializer()
    sess = tf.Session(config = config)
    sess.run(init)
    saver.restore(sess,PRETRAIN_MODEL_NAME)    
    
    def test():
        ACC = 0
        for i in range(lenn_t):
            test_batch = TestData[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            test_label = TestLabels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            ACC+=sess.run(acc,feed_dict={batch:test_batch,label:test_label,keep_prop:1.0})
        
        return ACC/lenn_t
    
    '''Begin Active Learning!'''
    log_file = open('./simple_log/'+Experiment_NAME+'_ALRA.txt','w')
    pretrain_accuracy = test()
    print 'the pre train model accuracy is: ',pretrain_accuracy
    
    log_file.write("the pretrain model acc is " + str(pretrain_accuracy))
    log_file.write('\n')
    oracle_idx = np.arange(len(OracleData))
    for a in range(ACTIVE_TIME):
        shuffle(oracle_idx)
        tag_queue = oracle_idx[0:QUEUE_SIZE]
        oracle_idx = oracle_idx[QUEUE_SIZE:]
        
        if a == 0:
            TrainData = OracleData[tag_queue]
            TrainLabels = OracleLabels[tag_queue]
        else:
            TrainData = np.concatenate((TrainData,OracleData[tag_queue]))
            TrainLabels = np.concatenate((TrainLabels, OracleLabels[tag_queue]))
            
        train_queue = np.arange(len(TrainData))
        best = 0
        for i in range(EPOCH_all):
            shuffle(train_queue)
            for j in range(len(TrainData)/BATCH_SIZE):
                trainbatch = TrainData[train_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
                trainlabels = TrainLabels[train_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
                sess.run(opt, feed_dict = {batch:trainbatch, label:trainlabels,keep_prop:0.5})
            accuracy = test()
            print 'the ',a+1, 'time acmodel acc is:', accuracy
            if accuracy > best:
                best = accuracy
                saver.save(sess,MODEL_PATH+Experiment_NAME+'_ALRA_'+str(a+1)+'.ckpt')
                print 'the ',a+1,' time acmodel best acc is: ', best
        log_file.write("the " + str(a+1) + " time acmodel best acc is " + str(best))
        log_file.write("\n")
    log_file.close()
    return 

#############################################################################################################################################
############################################################################################################################################
#Active learning with high entropy selection strategy
def entropy_active():
    #load data
    TrainData, TrainLabels, OracleData,  OracleLabels,\
    TestData, TestLabels, _, _, lenn_s, lenn_t, _ = load_process()
    
    batch = tf.placeholder(tf.float32,[None,unit.H,unit.W,unit.Channel])
    label = tf.placeholder(tf.uint8,[None])
    keep_prop = tf.placeholder(tf.float32)
    
    feature = SharePart(batch,keep_prop)
    result = MissionPart(feature)
    loss = SoftmaxWithLoss(result,label)
    acc = Test(result,label)
    opt = train_net(loss)
    
    softmax = tf.nn.softmax(result)
    entropy = tf.reduce_sum(-softmax * tf.log(softmax),1)
    predict_class = tf.cast(tf.argmax(softmax,axis = 1),tf.uint8)
    
    saver = tf.train.Saver(max_to_keep = ACTIVE_TIME)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    config.gpu_options.allow_growth = False
    init = tf.global_variables_initializer()
    sess = tf.Session(config = config)
    sess.run(init)
    saver.restore(sess,PRETRAIN_MODEL_NAME)    
    
    def test():
        ACC = 0
        for i in range(lenn_t):
            test_batch = TestData[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            test_label = TestLabels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            ACC+=sess.run(acc,feed_dict={batch:test_batch,label:test_label,keep_prop:1.0})
        
        return ACC/lenn_t
    
    '''Begin Active Learning!'''
    log_file = open('./simple_log/'+Experiment_NAME+'_ALST.txt','w')
    pretrain_accuracy = test()
    print 'the pre train model accuracy is : ', pretrain_accuracy
    log_file.write("the pre train model accuracy is " + str(pretrain_accuracy))
    log_file.write("\n")
   
#active learning   
    for a in range(ACTIVE_TIME):
        oracle_idx = np.arange(len(OracleData))
        oracle_que = []
		#calculate the entropy of each sample
        for i in oracle_idx:
            candidate_entropy = sess.run(entropy, feed_dict={batch:unit.changeshape_1(OracleData[i]),keep_prop:1.0})
            candidate_predict = sess.run(predict_class, feed_dict={batch:unit.changeshape_1(OracleData[i]),keep_prop:1.0})
            oracle_que.append((i,candidate_entropy[0],candidate_predict[0]))
        oracle_que = sorted(oracle_que, key = lambda candidate:candidate[1], reverse = True)
		#oracle_que contians 3 veriable, [image id; entropy; predicted label]
	#temp saves every entropy under every class and in every class entropies are sorted decreasingly
        temp = {}
        tag_queue = []
        for k in range(CLASS_NUM):
            temp[str(k)] = []
        for k in range(len(oracle_que)):
            temp[str(oracle_que[k][2])].append(oracle_que[k])
        for k in temp:
            temp[k] = sorted(temp[k], key=lambda x:x[1], reverse=True)
        #shuffle in class axis and pick utill QUEUE_SIZE
        idx = 0
        temp_class = 0
        temp_order = range(CLASS_NUM)
        shuffle(temp_order)
        while(idx<QUEUE_SIZE):
            if len(temp[str(temp_order[temp_class])]) != 0:
                tag_queue.append(temp[str(temp_order[temp_class])].pop(0)[0])
                idx += 1
                temp_class = (temp_class+1)%(CLASS_NUM)
            else:
                temp_class = (temp_class+1)%(CLASS_NUM)

        if a == 0 : 
            TrainData = OracleData[tag_queue]
            TrainLabels = OracleLabels[tag_queue]
            np.delete(OracleData, tag_queue), np.delete(OracleLabels,tag_queue)              #labeled samples are deleted for unlabeled pool
        else:
            TrainData = np.concatenate((TrainData,OracleData[tag_queue]))
            TrainLabels = np.concatenate((TrainLabels, OracleLabels[tag_queue]))
            np.delete(OracleData, tag_queue), np.delete(OracleLabels,tag_queue)
        
        train_queue = np.arange(len(TrainData))
        best = 0
        for i in range(EPOCH_all):
            shuffle(train_queue)
            for j in range(len(TrainData)/BATCH_SIZE):
                trainbatch = TrainData[train_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
                trainlabels = TrainLabels[train_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
                sess.run(opt, feed_dict = {batch:trainbatch, label:trainlabels,keep_prop:0.5})
            accuracy = test()
            print 'the ',a+1, 'time acmodel acc is:', accuracy
            if accuracy > best:
                best = accuracy
                saver.save(sess,MODEL_PATH+Experiment_NAME+'_ALST_'+str(a+1)+'.ckpt')
                print 'the ',a+1,' time acmodel best acc is: ', best
        log_file.write("the " + str(a+1) + "time acmodel best acc is " + str(best))
        log_file.write("\n")
    return 
	
#################################################################################################################
###############################################################################################################
#Bi-direction active learning with high entropy samples and high confidence samples.
def bidirectional_active():
#load data
    TrainData, TrainLabels, OracleData,  OracleLabels,\
    TestData, TestLabels, _, _, lenn_s, lenn_t, _ = load_process()
    
    batch = tf.placeholder(tf.float32,[None,unit.H,unit.W,unit.Channel])
    label = tf.placeholder(tf.uint8,[None])
    keep_prop = tf.placeholder(tf.float32)
    
    feature = SharePart(batch,keep_prop)
    result = MissionPart(feature)
    loss = SoftmaxWithLoss(result,label)
    acc = Test(result,label)
    opt = train_net(loss)
    
    softmax = tf.nn.softmax(result)
    entropy = tf.reduce_sum(-softmax * tf.log(softmax),1)
    predict_class = tf.cast(tf.argmax(softmax,axis = 1),tf.uint8)
    
    saver = tf.train.Saver(max_to_keep = ACTIVE_TIME)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    config.gpu_options.allow_growth = False
    init = tf.global_variables_initializer()
    sess = tf.Session(config = config)
    sess.run(init)
    
    saver.restore(sess,PRETRAIN_MODEL_NAME)    
    
    def test():
        ACC = 0
        for i in range(lenn_t):
            test_batch = TestData[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            test_label = TestLabels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            ACC+=sess.run(acc,feed_dict={batch:test_batch,label:test_label,keep_prop:1.0})
        
        return ACC/lenn_t
    
    '''Begin Active Learning!'''
    log_file = open('./simple_log/'+Experiment_NAME+'_ALBT.txt','w')
    pretrain_accuracy = test()
    print 'the pre train model accuracy is : ', pretrain_accuracy
    log_file.write("the pre train model accuracy is " + str(pretrain_accuracy))
    log_file.write("\n")

    
    for a in range(ACTIVE_TIME):
        oracle_idx = np.arange(len(OracleData))
        oracle_que = []
        for i in oracle_idx:
            candidate_entropy = sess.run(entropy, feed_dict={batch:unit.changeshape_1(OracleData[i]),keep_prop:1.0})
            candidate_predict = sess.run(predict_class, feed_dict={batch:unit.changeshape_1(OracleData[i]),keep_prop:1.0})
            oracle_que.append((i,candidate_entropy[0],candidate_predict[0]))
        oracle_que = sorted(oracle_que, key = lambda candidate:candidate[1], reverse = True)
        temp = {}
        tag_queue = []
        tag_queue2 = []
        tag_queue2_labels = []
        for k in range(CLASS_NUM):
            temp[str(k)] = []
        for k in range(len(oracle_que)):
            temp[str(oracle_que[k][2])].append(oracle_que[k])
        for k in temp:
            temp[k] = sorted(temp[k], key=lambda x:x[1], reverse=True)
            
        idx = 0
        temp_class = 0
        temp_order = range(CLASS_NUM)
        shuffle(temp_order)
        while(idx<QUEUE_SIZE):
            if len(temp[str(temp_order[temp_class])]) != 0:
                tag_queue.append(temp[str(temp_order[temp_class])].pop(0)[0])
                idx += 1
                temp_class = (temp_class+1)%(CLASS_NUM)
            else:
                temp_class = (temp_class+1)%(CLASS_NUM)
        idx = 0 
        temp_class = 0
        while(idx< GOOD_SIZE ):
            if len(temp[str(temp_order[temp_class])]) != 0:
                tag_temporary = temp[str(temp_order[temp_class])].pop()
                tag_queue2.append(tag_temporary[0])
                tag_queue2_labels.append(tag_temporary[2])
                idx += 1
                temp_class = (temp_class + 1)%(CLASS_NUM)
            else:
                temp_class = (temp_class + 1)%(CLASS_NUM)
                
#########################################'''not put back, x_train+x_oracle'''         
                
#        TrainData = np.concatenate((TrainData,OracleData[tag_queue]))
#        TrainData = np.concatenate((TrainData,OracleData[tag_queue2]))
#        TrainLabels = np.concatenate((TrainLabels, OracleLabels[tag_queue]))
#        TrainLabels = np.concatenate((TrainLabels, np.array(tag_queue2_labels)))      
#        tag_queue2_rlabels = OracleLabels[tag_queue2]
#        np.delete(OracleData, tag_queue + tag_queue2), np.delete(OracleLabels,tag_queue + tag_queue2)
        
###############################################''' put back , x_train+x_oracle'''
#        TrainData0 =  dc(TrainData)
#        TrainLabels0 = dc(TrainLabels)
#        TrainData0 = np.concatenate((TrainData0,OracleData[tag_queue]))
#        TrainLabels0 = np.concatenate((TrainLabels0,OracleLabels[tag_queue]))
#        TrainData = np.concatenate((TrainData0, OracleData[tag_queue2]))
#        TrainLabels = np.concatenate((TrainLabels0, OracleLabels[tag_queue2]))
#        tag_queue2_rlabels = OracleLabels[tag_queue2]
#        np.delete(OracleData, tag_queue), np.delete(OracleLabels,tag_queue)


################################################not put back,x_oracle   
        if a == 0:
            TrainData = np.concatenate((OracleData[tag_queue],OracleData[tag_queue2]))
            TrainLabels = np.concatenate((OracleLabels[tag_queue], np.array(tag_queue2_labels)))
        else:
            TrainData = np.concatenate((TrainData,OracleData[tag_queue]))
            TrainData = np.concatenate((TrainData,OracleData[tag_queue2]))
            TrainLabels = np.concatenate((TrainLabels, OracleLabels[tag_queue]))
            TrainLabels = np.concatenate((TrainLabels, np.array(tag_queue2_labels)))  
        tag_queue2_rlabels = OracleLabels[tag_queue2]
        np.delete(OracleData, tag_queue + tag_queue2), np.delete(OracleLabels,tag_queue + tag_queue2)
#############################################################################################################        
        
        train_queue = np.arange(len(TrainData))
        best = 0
        for i in range(EPOCH_all):
            shuffle(train_queue)
            for j in range(len(TrainData)/BATCH_SIZE):
                trainbatch = TrainData[train_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
                trainlabels = TrainLabels[train_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
                sess.run(opt, feed_dict = {batch:trainbatch, label:trainlabels,keep_prop:0.5})
            accuracy = test()
            print 'the ',a+1, 'time acmodel acc is:', accuracy
            if accuracy > best:
                best = accuracy
                saver.save(sess,MODEL_PATH+Experiment_NAME+'_ALBT_'+str(a+1)+'.ckpt')
                print 'the ',a+1,' time acmodel best acc is: ', best
        cnn_acc = np.float(np.sum(np.equal(tag_queue2_rlabels,np.array(tag_queue2_labels)))) / GOOD_SIZE
        log_file.write("the " + str(a+1) + "time acmodel best acc is " + str(best))
        log_file.write("\n")
        log_file.write("the " + str(a+1) + "time cnn_que acc is " + str(cnn_acc))
        log_file.write("\n")
    log_file.close()
    return 
                
###################################################################################################################
##################################################################################################################
#active learning with strategy taking ground truth label as model prediction用真实标签代替模型标签的双向学习方法
def bidirectional_active_expert():
    TrainData, TrainLabels, OracleData,  OracleLabels,\
    TestData, TestLabels, _, _, lenn_s, lenn_t, _ = load_process()
    
    batch = tf.placeholder(tf.float32,[None,unit.H,unit.W,unit.Channel])
    label = tf.placeholder(tf.uint8,[None])
    keep_prop = tf.placeholder(tf.float32)
    
    feature = SharePart(batch,keep_prop)
    result = MissionPart(feature)
    loss = SoftmaxWithLoss(result,label)
    acc = Test(result,label)
    opt = train_net(loss)
    
    softmax = tf.nn.softmax(result)
    entropy = tf.reduce_sum(-softmax * tf.log(softmax),1)
    predict_class = tf.cast(tf.argmax(softmax,axis = 1),tf.uint8)
    
    saver = tf.train.Saver(max_to_keep = ACTIVE_TIME)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    config.gpu_options.allow_growth = False
    init = tf.global_variables_initializer()
    sess = tf.Session(config = config)
    sess.run(init)
    
    saver.restore(sess,PRETRAIN_MODEL_NAME)    
    
    def test():
        ACC = 0
        for i in range(lenn_t):
            test_batch = TestData[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            test_label = TestLabels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            ACC+=sess.run(acc,feed_dict={batch:test_batch,label:test_label,keep_prop:1.0})
        
        return ACC/lenn_t
    
    '''Begin Active Learning!'''
    log_file = open('./simple_log/'+Experiment_NAME+'_ALBT_man.txt','w')
    pretrain_accuracy = test()
    print 'the pre train model accuracy is : ', pretrain_accuracy
    log_file.write("the pre train model accuracy is " + str(pretrain_accuracy))
    log_file.write("\n")

    
    for a in range(ACTIVE_TIME):
        oracle_idx = np.arange(len(OracleData))
        oracle_que = []
        for i in oracle_idx:
            candidate_entropy = sess.run(entropy, feed_dict={batch:unit.changeshape_1(OracleData[i]),keep_prop:1.0})
            candidate_predict = sess.run(predict_class, feed_dict={batch:unit.changeshape_1(OracleData[i]),keep_prop:1.0})
            oracle_que.append((i,candidate_entropy[0],candidate_predict[0]))
        oracle_que = sorted(oracle_que, key = lambda candidate:candidate[1], reverse = True)
        temp = {}
        tag_queue = []
        tag_queue2 = []
        tag_queue2_labels = []
        for k in range(CLASS_NUM):
            temp[str(k)] = []
        for k in range(len(oracle_que)):
            temp[str(oracle_que[k][2])].append(oracle_que[k])
        for k in temp:
            temp[k] = sorted(temp[k], key=lambda x:x[1], reverse=True)
            
        idx = 0
        temp_class = 0
        temp_order = range(CLASS_NUM)
        shuffle(temp_order)
        while(idx<QUEUE_SIZE):
            if len(temp[str(temp_order[temp_class])]) != 0:
                tag_queue.append(temp[str(temp_order[temp_class])].pop(0)[0])
                idx += 1
                temp_class = (temp_class+1)%(CLASS_NUM)
            else:
                temp_class = (temp_class+1)%(CLASS_NUM)
        idx = 0 
        temp_class = 0
        while(idx< GOOD_SIZE ):
            if len(temp[str(temp_order[temp_class])]) != 0:
                tag_temporary = temp[str(temp_order[temp_class])].pop()
                tag_queue2.append(tag_temporary[0])
                tag_queue2_labels.append(tag_temporary[2])
                idx += 1
                temp_class = (temp_class + 1)%(CLASS_NUM)
            else:
                temp_class = (temp_class + 1)%(CLASS_NUM)
                

##################################not put back ,x_train+x_oracle                
#        TrainData = np.concatenate((TrainData,OracleData[tag_queue]))
#        TrainData = np.concatenate((TrainData,OracleData[tag_queue2]))
#        TrainLabels = np.concatenate((TrainLabels, OracleLabels[tag_queue]))
#        TrainLabels = np.concatenate((TrainLabels, OracleLabels[tag_queue2])) 
#        tag_queue2_rlabels = OracleLabels[tag_queue2]
#        np.delete(OracleData, tag_queue + tag_queue2), np.delete(OracleLabels,tag_queue + tag_queue2)


##########################################put back, x_train+x_oracle        
#        TrainData0 =  dc(TrainData)
#        TrainLabels0 = dc(TrainLabels)
#        TrainData0 = np.concatenate((TrainData0,OracleData[tag_queue]))
#        TrainLabels0 = np.concatenate((TrainLabels0,OracleLabels[tag_queue]))
#        TrainData = np.concatenate((TrainData0, OracleData[tag_queue2]))
#        TrainLabels = np.concatenate((TrainLabels0, OracleLabels[tag_queue2]))
#        tag_queue2_rlabels = OracleLabels[tag_queue2]
#        np.delete(OracleData, tag_queue), np.delete(OracleLabels,tag_queue)
        

################################################not put back,x_oracle
        if a == 0:
            TrainData = np.concatenate((OracleData[tag_queue],OracleData[tag_queue2]))
            TrainLabels = np.concatenate((OracleLabels[tag_queue], np.array(tag_queue2_labels)))
        else:
            TrainData = np.concatenate((TrainData,OracleData[tag_queue]))
            TrainData = np.concatenate((TrainData,OracleData[tag_queue2]))
            TrainLabels = np.concatenate((TrainLabels, OracleLabels[tag_queue]))
            TrainLabels = np.concatenate((TrainLabels, np.array(tag_queue2_labels)))  
        tag_queue2_rlabels = OracleLabels[tag_queue2]
        np.delete(OracleData, tag_queue + tag_queue2), np.delete(OracleLabels,tag_queue + tag_queue2)

        
        train_queue = np.arange(len(TrainData))
        best = 0
        for i in range(EPOCH_all):
            shuffle(train_queue)
            for j in range(len(TrainData)/BATCH_SIZE):
                trainbatch = TrainData[train_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
                trainlabels = TrainLabels[train_queue[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]
                sess.run(opt, feed_dict = {batch:trainbatch, label:trainlabels,keep_prop:0.5})
            accuracy = test()
            print 'the ',a+1, 'time acmodel acc is:', accuracy
            if accuracy > best:
                best = accuracy
                saver.save(sess,MODEL_PATH+Experiment_NAME+'_ALBT_man_'+str(a+1)+'.ckpt')
                print 'the ',a+1,' time acmodel best acc is: ', best
        cnn_acc = np.float(np.sum(np.equal(tag_queue2_rlabels,np.array(tag_queue2_labels)))) / GOOD_SIZE
        log_file.write("the " + str(a+1) + "time acmodel best acc is " + str(best))
        log_file.write("\n")
        log_file.write("the " + str(a+1) + "time cnn_que acc is " + str(cnn_acc))
        log_file.write("\n")
    log_file.close()
    return


#############################################################################################
#test and generate confusion matrixes/precision/recall/f1 for each model
def Test_model_process():
    file_test = FILE_PATH + Experiment_NAME + '_x_test.txt'
#    file_test2 = FILE_PATH + Experiment_NAME + '_x_oracle.txt'
    ac_methods = ['_ALBT_','_ALBT_man_','_ALST_','_ALRA_']  #_ALBT_,_ALST_,_ALRA_
#    OList = unit.LoadCarTxT(file_test2)
    TestList = unit.LoadCarTxT(file_test)
#    OData, OLabels = unit.Getlist(OList)
    TestData, TestLabels = unit.Getlist(TestList)
#    TestData = np.concatenate((TestData,OData))
#    TestLabels = np.concatenate((TestLabels, OLabels))
    
    batch = tf.placeholder(tf.float32,[None,unit.H,unit.W,unit.Channel])
    label = tf.placeholder(tf.uint8,[None])
    keep_prop = tf.placeholder(tf.float32)
    
    feature = SharePart(batch,keep_prop)
    result = MissionPart(feature)
    loss = SoftmaxWithLoss(result,label)
    acc = Test(result,label)
    opt = train_net(loss)
    
    softmax = tf.nn.softmax(result)
    entropy = tf.reduce_sum(-softmax * tf.log(softmax),1)
    predict_class = tf.cast(tf.argmax(result,axis = 1),tf.uint8)
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = False
    init = tf.global_variables_initializer()
    sess = tf.Session(config = config)
    sess.run(init)
      
#    confunsion_matix = []
#    file_report = open(MODEL_PATH+'confusion_matrix/'+Experiment_NAME+'_pre_'+'report.txt','w')
#    recall = []
#    precision = []
#    f1 = []
#    support = []
#    accuracy = 0
#    c_m_t = np.zeros([CLASS_NUM,CLASS_NUM])
#    saver.restore(sess,MODEL_PATH+PRETRAIN_MODEL_NAME)
#    for i in range(len(TestData)):
#        predict_label = sess.run(predict_class,feed_dict = { batch:unit.changeshape_1(TestData[i]),keep_prop:1.0})
#        c_m_t[TestLabels[i],predict_label] += 1
#    confunsion_matix.append(c_m_t)
#    np.save(MODEL_PATH+'confusion_matrix/'+Experiment_NAME+'_pre'+'.npy',c_m_t)
#    file_report.write(" stage " + "confusion_matrix with testsets\n" )
#    file_report.write("          "+"precision".rjust(10,)+"recall".rjust(10,)+"f1-score".rjust(10,)+"support".rjust(10,)+'\n')
#    plotconfusion(c_m_t,MODEL_PATH+'confusion_matrix/'+Experiment_NAME+'_pre',CLASS_NUM)
#    for i in range(CLASS_NUM):
#        accuracy += c_m_t[i,i]
#        try:
#            recall.append(round(c_m_t[i,i]/np.sum(c_m_t[i]),3))
#        except:
#            recall.apprend(round(0,3))
#        try:
#            precision.append(round(c_m_t[i,i]/np.sum(c_m_t[:,i]),3))
#        except:
#            precision.append(round(0,3))
#        try:
#            f1.append(round(2*recall[i]*precision[i]/(recall[i]+precision[i]),3))
#        except:
#            f1.append(round(0,3))
#        support.append(np.sum(c_m_t[i]))
#        file_report.write(str(i).rjust(10,)+str(precision[i]).rjust(10,)+str(recall[i]).rjust(10,)+str(f1[i]).rjust(10,)+str(support[i]).rjust(10,)+'\n')
#    try:
#        recall_avg = round(np.sum(np.array(recall))/CLASS_NUM,3)
#    except:
#        recall_avg = 0
#    try:
#        precision_avg = round(np.sum(np.array(precision))/CLASS_NUM,3)
#    except:
#        precision_avg = 0
#    try:
#        f1_avg = round(np.sum(np.array(f1))/CLASS_NUM,3)
#    except:
#        f1_avg = 0
#    support_num = np.sum(np.array(support))
#    accuracy = round(accuracy/support_num,5)
#    file_report.write("average".rjust(10,)+str(precision_avg).rjust(10,)+str(recall_avg).rjust(10,)+str(f1_avg).rjust(10,)+str(support_num).rjust(10,)+'\n')
#    file_report.write(" stage acc is " +str(accuracy))
#    file_report.write("\n\n\n\n")
#    file_report.close()

#
    for ac_method in ac_methods:
        file_report = open('./confusion_matrix/'+Experiment_NAME+ac_method+'report.txt','w')
        for a in range(ACTIVE_TIME):
            recall = []
            precision = []
            f1 = []
            support = []
            accuracy = 0
            c_m_t = np.zeros([CLASS_NUM,CLASS_NUM])
            saver.restore(sess,MODEL_PATH+Experiment_NAME+ac_method+str(a+1)+'.ckpt')
            for i in range(len(TestData)):
                predict_label = sess.run(predict_class,feed_dict = { batch:unit.changeshape_1(TestData[i]),keep_prop:1.0})
                c_m_t[TestLabels[i],predict_label] += 1
            confunsion_matix.append(c_m_t)
            np.save('./confusion_matrix/'+Experiment_NAME+ac_method+str(a+1)+'.npy',c_m_t)
#            np.save('./confusion_matrix/'+Experiment_NAME+'_pre'+'.npy',c_m_t)
            file_report.write(str(a+1) + " stage " + "confusion_matrix with testsets\n" )
            file_report.write("          "+"precision".rjust(10,)+"recall".rjust(10,)+"f1-score".rjust(10,)+"support".rjust(10,)+'\n')
            plotconfusion(c_m_t,'./confusion_matrix/'+Experiment_NAME+ac_method+str(a+1),CLASS_NUM)
#            plotconfusion(c_m_t,'./confusion_matrix/'+Experiment_NAME+'_pre',CLASS_NUM)
            for i in range(CLASS_NUM):
                accuracy += c_m_t[i,i]
                try:
                    recall.append(round(c_m_t[i,i]/np.sum(c_m_t[i]),3))
                except:
                    recall.apprend(round(0,3))
                try:
                    precision.append(round(c_m_t[i,i]/np.sum(c_m_t[:,i]),3))
                except:
                    precision.append(round(0,3))
                try:
                    f1.append(round(2*recall[i]*precision[i]/(recall[i]+precision[i]),3))
                except:
                    f1.append(round(0,3))
                support.append(np.sum(c_m_t[i]))
                file_report.write(str(i).rjust(10,)+str(precision[i]).rjust(10,)+str(recall[i]).rjust(10,)+str(f1[i]).rjust(10,)+str(support[i]).rjust(10,)+'\n')
            try:
                recall_avg = round(np.sum(np.array(recall))/CLASS_NUM,3)
            except:
                recall_avg = 0
            try:
                precision_avg = round(np.sum(np.array(precision))/CLASS_NUM,3)
            except:
                precision_avg = 0
            try:
                f1_avg = round(np.sum(np.array(f1))/CLASS_NUM,3)
            except:
                f1_avg = 0
            support_num = np.sum(np.array(support))
            accuracy = round(accuracy/support_num,5)
            file_report.write("average".rjust(10,)+str(precision_avg).rjust(10,)+str(recall_avg).rjust(10,)+str(f1_avg).rjust(10,)+str(support_num).rjust(10,)+'\n')
            file_report.write(str(a+1) + " stage acc is " +str(accuracy))
            file_report.write("\n\n\n\n")
        file_report.close()

    
    return 

            
if __name__=='__main__':
#    data_process()
#    pretrain()
#    random_active()
#    entropy_active()
#    bidirectional_active()
#    bidirectional_active_expert()
    Test_model_process()
