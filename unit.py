# -*- coding: utf-8 -*-

import os
from random import shuffle
from PIL import Image
import numpy as np
import itertools
H = 227
W = 227
Channel = 3

#输入id,label,dir。会生成一个list，每个元素为[路径，标签]，将dir内图片全部取出，注意这里只是输出路径和标签，而不是图片本身
def get_list(car_id,car_label,sample_dir):
    result = []
    for k in list(range(len(car_id))):
        for dirpath,dirnames,filenames in os.walk(sample_dir+str(car_id[k])):
            for filename in filenames:
                path = os.path.join(dirpath,filename)
                result.append((path,car_label[k]))
#    shuffle(result)
    return result
#输入同上，输出相同，加入的totalnum可以控制取出多少张图片，一个dir内取出totalnum张图片。
def get_list2(car_id,car_label,sample_dir,totalnum):
    result = []
    for k in list(range(len(car_id))):
        for dirpath,dirnames,filenames in os.walk(sample_dir+str(car_id[k])):
            iii = 0
            shuffle(filenames)
            for filename in filenames:
                path = os.path.join(dirpath,filename)
                result.append((path,car_label[k]))
                iii += 1
                if iii==totalnum:
                    break
    shuffle(result)
    return result

#totalnum控制初始训练集数目，or_res_t中那一行的比例确定的是未标注样本池的规模，剩余的为测试集规模，一共返回三个list，初始训练集，未标注样本集，测试集。
def GetListTvT(car_id,car_label,sample_dir,totalnum):
    TrainResult, OracleResult, TestResult = [],[],[]
    for k in list(range(len(car_id))):
        for DirPath, DirNames, FileNames in os.walk(sample_dir + str(car_id[k])):
            shuffle(FileNames)
            tr_res_t, or_res_t, te_res_t = [],[],[]
            tr_res_t.append(FileNames[0:totalnum])
            FN_remain = FileNames
            or_res_t.append(FN_remain[totalnum:len(FN_remain)*2/3])
            te_res_t.append(FN_remain[len(FN_remain)*2/3:])
            tr_res_t, or_res_t, te_res_t = list(itertools.chain(*tr_res_t)), list(itertools.chain(*or_res_t)), list(itertools.chain(*te_res_t))
            for i in tr_res_t:
                path = os.path.join(DirPath, i)
                TrainResult.append((path,car_label[k]))
            for i in or_res_t:
                path = os.path.join(DirPath, i)
                OracleResult.append((path,car_label[k]))
            for i in te_res_t:
                path = os.path.join(DirPath, i)
                TestResult.append((path,car_label[k]))
    return TrainResult, OracleResult, TestResult

#只返回未标注样本集和测试集
def GetListTvT_carsdatatest(car_id,car_label,sample_dir):
    OracleResult, TestResult = [],[]
    for k in list(range(len(car_id))):
        for DirPath, DirNames, FileNames in os.walk(sample_dir + str(car_id[k])):
            shuffle(FileNames)
            or_res_t, te_res_t = [],[]
            FN_remain = FileNames
            or_res_t.append(FN_remain[0:len(FN_remain)*2/3])
            te_res_t.append(FN_remain[len(FN_remain)*2/3:])
            or_res_t, te_res_t = list(itertools.chain(*or_res_t)), list(itertools.chain(*te_res_t))
            for i in or_res_t:
                path = os.path.join(DirPath, i)
                OracleResult.append((path,car_label[k]))
            for i in te_res_t:
                path = os.path.join(DirPath, i)
                TestResult.append((path,car_label[k]))
    return  OracleResult, TestResult

#将初始训练集也加入了未标注样本池中，所以这里从0开始。返回两个List，trn这个就是未标注样本池，test这个就是测试集。
def trn_test_list(car_id,car_label,sample_dir):
    ''' select half of the dataset be trn set'''
    trn_dataset = []
    test_dataset = []
    for k in list(range(len(car_id))):
        temp_list = []
        for dirpath,dirnames,filenames in os.walk(sample_dir+str(car_id[k])):
            for filename in filenames:
                path = os.path.join(dirpath,filename)
                temp_list.append((path,car_label[k]))
#            shuffle(temp_list)
            num = len(temp_list)
            for i in temp_list[0:num*2/3]:
                trn_dataset.append(i)
            for i in temp_list[num*2/3:]:
                test_dataset.append(i)
    return trn_dataset,test_dataset

def load_oracle(oracle_list):
    trn_dataset = []
    test_dataset = []
    file = open(oracle_list,'r')
    temp_list = []
    car_id_label = file.readlines()
    for i in car_id_label:
        i2 = i.strip().strip("()'")
        car_id,car_label = i2.split(",")
        temp_list.append((car_id.strip("'"),int(car_label)))
    file.close()
    num = len(temp_list)
    for i in temp_list[0:num*2/3]:
        trn_dataset.append(i)
    for i in temp_list[num*2/3:]:
        test_dataset.append(i)
    return trn_dataset, test_dataset

#这个是根据之前生成的文件载入数据的函数，载入之前保存的list,[路径，标签]
def LoadCarTxT(FileName):
    file = open(FileName,'r')
    dataset = []
    CarID_Label = file.readlines()
    for i in CarID_Label:
        i2 = i.strip().strip("()'")
        car_id, car_label = i2.split(",")
        dataset.append((car_id.strip("'"),int(car_label)))
    file.close()
    return dataset

def GetBatch(sample_list,batchsize,idx):  

    batch = np.zeros([batchsize,H,W,Channel])
    label = []
    for i in range(batchsize):
        if ((idx + i) >= len(sample_list)):
            idx = 0
        img = Image.open(sample_list[idx+i][0])
        img = img.resize([H,W])
        img = np.array(img)
        batch[i,:,:,:] =  img
        label.append(sample_list[idx+i][1])
        
    batch = np.float32(batch)
    label = np.uint8(label)
    return batch,label

#将单个数据改成合适的格式，适用于测试时候用。
def changeshape_1(simple):
    return simple.reshape([-1,H,W,Channel])

#根据输入的路径直接输出数据和标签，返回的值为路径List中所有的数据，以及标签
def Getlist(sample_list):
    batch = np.zeros([len(sample_list),H,W,Channel])
    label = []
    for i in range(len(sample_list)):
        img = Image.open(sample_list[i][0])
        img = img.resize([H,W])
        img = np.array(img)
        batch[i,:,:,:] = img
        label.append(sample_list[i][1])
    
    batch = np.float32(batch)
    label = np.uint8(label)
    return batch, label
           
#source_dir = '/home/zhengh/carpictures/orderzheng/'
#car_id = [1,15,20,35,48,60,73,86,111,124,137,162,175,188,212,225,238,250,263,276]
#car_label = [0,4,8,15,16,17,18,19,1,2,3,5,6,7,9,10,11,12,13,14]
#a,b = trn_test_list(car_id,car_label,source_dir)  
