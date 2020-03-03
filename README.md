# calculate-precision-recall-f1score-roc-auc
Precision, recall, f1-score, AUC, loss, accuracy and ROC curve are often used in binary image recognition evaluation issue. The repository calculates the metrics based on the data of one epoch rather than one batch, which means the criteria is more reliable.  The program implements the calculation at the end of the training process and every epoch process through two versions independently on keras.

#中文说明
这个仓库的程序基于keras计算Precision, recall, f1-score, AUC, loss, accuracy值以及绘制ROC曲线。keras自带的Precision, recall, f1-score是基于
一个batch图片的结果，这样显然是不能描述整个训练数据集或者验证集的metrics。我在python库sklearn上写了一个程序基于整个数据集（one epoch)计算上述metrics。

本仓库的程序很简单，用一个程序文件即可实现功能。程序是通过跑一个具体的深度学习模型展示其计算过程。
四个程序中相同文件名的后缀.ipynb文件和.py文件是一样的程序，只是文件形式不同，.ipynb用jupyter打开，.py则是用spyder打开，建议使用.ipynb打开程序。
criteria_end_of_epoch是每一个epoch后计算一次训练集和验证集的metrics，且在所有epoch训练完成后绘制metrics-epoch图以观察训练过程中metrics的变化情况。
criteria_end_of_train是在训练完成后计算训练集和验证集的metrics，因为对于大型的图片数据集，如果每一个epoch都计算metrics，那么将消耗很多计算资源并且训练时间会很长。

程序需要设置的参数，比如：训练集目录、验证集目录等等，直接在程序文件内操作。绝大多数参数是在.ipynb的第二个单元格中修改，其他修改参数可看注释，注释已明确说明。
另外因为metrics计算是基于具体深度学习模型的，我简单的搭建一个模型作为示范，你们应用本metrics程序到自己模型上时，可以通过load_model的方式加载模型或者直接在设计模型的单元格内敲模型设计的代码。

#English introduction
The program for this repository calculates precision, recall, f1-score, auc, loss, accuracy values as well as draws the roc curve based on keras. Keras's own precion, recall, and f1-score are based on the results of one batch image, so it's clear that the metrics can't decribe the performance of entire training or validation data set. I wrote a program based on the python library, sklearn, to calculate the above metrics based on the entire data set. 

The program of repository is very simple, with a program file to achieve the function. The program is to demonstrate its calculation process by running a specific deep learning model. 
The same file name in the four programs with the suffix of  'ipynb' file and 'py' file are the same programs. Just the file form is different. 'ipynb' opens it with jupyter and 'py' is opened with spyder. I recommend you use the 'ipynb' file.  
"Criteria_end_of_epoch" is a method for calculating the training set and validation set after each epoch, and drawing the "method-epoch" diagram after all epoch training is completed to observe the changes of the method during the training process. 
"Criteria_end_of_train" computes the metric for the training set and validation set after the training is completed, because for large picture datasets, if each epoch computes metric, it consumes a lot of computational resources and the training time is long. 

Procedures need to set the parameters, such as: "training set directory", "validation set directory" and so on, operation this parameters directly in the program file. The majority of parameters are in 'ipynb' file's second cell, other parameters can be seen in the annotations, the annotations have been clearly stated. 
In addition, because the metric calculation is based on the specific deep learning model, I simply build a model as a demonstration. When you apply this metrics program to your own model, you can load the model in the way of "load_model" or type the code in the cell of tdesigning model directly. 
