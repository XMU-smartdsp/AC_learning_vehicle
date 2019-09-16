# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:03:56 2019

@author: Design
"""


from sklearn.metrics import confusion_matrix    # generate confusion matrix
import matplotlib.pyplot as plt    # plot lib
import matplotlib as mat
import numpy as np
#import tensorflow as tf

zhfont1 = mat.font_manager.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')

def plot_confusion_matrix(cm, title):
    cm_o = cm
    plt.rcParams['figure.figsize'] = (12.0, 10.0)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # normalization
    plt.imshow(cm, interpolation='nearest')    # show image
    plt.title(title,fontproperties =zhfont1, fontsize=40)    # caption
    plt.colorbar()
    num_local = np.array(range(20))    
    plt.xticks(num_local, num_local, rotation=90)    # label in x axis
    plt.yticks(num_local, num_local)    # label in y axis上
    plt.ylabel('ground true',fontproperties =zhfont1, fontsize=40)   #name of y axis 
    plt.xlabel('prediction',fontproperties =zhfont1, fontsize=40)   #name of x axis
    for i in num_local:
#        for j in num_local:
        plt.text(i, i, str('%.2f' % (cm[i, i])), va='center', ha='center')

cm1 = np.load('./confusion_matrix/comps_sv_20(3)_3_pre.npy')
plot_confusion_matrix(cm1, "pre-train")#image name
plt.savefig('./conf_pic/1.png', format='png')  #file name
plt.show()
