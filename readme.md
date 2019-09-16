ENGLISH
#1.AC_process1.py：
* main script of program。
* if you are from the very begining of the experiment, please run following with main function. 1. use data_process() to generate data list. 2. use pretrain() to initialize model. 3. run 4 active learning method. 4. after all done, run Test_model_process() to generate confusion matrix and other value.

* Experiment_NAME is quite important. If you already generated data and initialized model, step 1 and 2 could be ignored and you can run active learning directly.


#2.file：
* performance index report is generate in name :Experiment_NAME + method_name + report, saved under confusion_matrix directory. txt file

* precision,recall,f1-score generated in Test_model_process(). txt file

* simple confusion matrix is generated in name:Experiment_NAME + method_name + stage, saved under confusion_matrix directory. black-white color. png file


* confusion matrix is generated in name: Experiment_NAME + method_name + stage saved under confusion_matrix directory. Matrix. npy file


* simple log is generated in name: Experiment_NAME + method_name, saved under simple_log directory. Simply saved precision in every stage. For bi-direction entropy selection, high confidence sample precision is also saved. txt file.


* Plot_conf.py:
plot confusion matrix

#3.Unit.py:
load data

#4.layer.py:
network layer function

#5.data format
save image in 1,2,3....directory. Every directory contains 1 class cars image.





CHINESE

#1.AC_process1.py注意事项：
* 程序主脚本。
* 整个实验如果是从头开始，应当按照主函数流程进行。首先运行data\_process()生成数据目录，然后运行pretrain()生成初始模型，然后运行四个主动学习过程，生成阶段模型所有模型生成完毕后，运行Test_model_process()来生成混淆矩阵以及各种相关值。

* 其中Experiment_NAME的设置很关键，如果你已经生成好了数据以及初始模型，则可以直接运行主动过程，而无需全部从新来过。


#2.生成文件一览：
* 指标report，命名规则：实验名+方法名+report，存放在confusion_matrix目录下。txt文件

* 执行Test_model_process（）时生成的关于precision,recall,f1-score的报告，记录了每个阶段的每个类目的三项指标。txt文件

* 简易混淆矩阵，命名规则：实验名+方法名+阶段，存放在confusion_matrix目录下。颜色是黑白的。png文件


* 混淆矩阵，命名规则：实验名+方法名+阶段，存放在confusion_matrix目录下。数字形式，类目数乘类目数的一个数组矩阵。npy文件


* 简易日志，命名规则：实验名+方法，存放在simple_log文件夹下。简单记录了每一个阶段的准确率，对于双向算法同时记录了高置信度样本准确率。txt文件


* Plot_conf.py 注意事项：
绘制混淆矩阵的脚本。载入数字混淆矩阵后绘图即可。

#3.Unit.py注意事项：
一些读取数据时的函数,具体看注释即可。

#4.layer.py注意事项：
网络层函数。

#5.数据格式
按1，2，3，4，5...的目录名存放车型图片，每个目录下面为一个类别的车辆。
