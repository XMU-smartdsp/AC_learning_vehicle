# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:03:56 2019

@author: Design
"""


from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import matplotlib as mat
import numpy as np
#import tensorflow as tf

zhfont1 = mat.font_manager.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')

def plot_confusion_matrix(cm, title):
    cm_o = cm
    plt.rcParams['figure.figsize'] = (12.0, 10.0)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title,fontproperties =zhfont1, fontsize=40)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(20))    
    plt.xticks(num_local, num_local, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, num_local)    # 将标签印在y轴坐标上
    plt.ylabel('真实标签',fontproperties =zhfont1, fontsize=40)   #y轴名称 
    plt.xlabel('预测标签',fontproperties =zhfont1, fontsize=40)   #x轴名称
    for i in num_local:
#        for j in num_local:
        plt.text(i, i, str('%.2f' % (cm[i, i])), va='center', ha='center')

cm1 = np.load('./confusion_matrix/comps_sv_20(3)_3_pre.npy')
plot_confusion_matrix(cm1, "初始阶段")#这里输入的是图片的名称
plt.savefig('./conf_pic/1.png', format='png')  #保存文件名称
plt.show()
