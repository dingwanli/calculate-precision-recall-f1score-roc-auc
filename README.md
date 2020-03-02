# calculate-precision-recall-f1score-roc-auc
Precision, recall, f1-score, AUC, loss, accuracy and ROC curve are often used in binary image recognition evaluation issue. The repository calculates the metrics based on the data of one epoch rather than one batch, which means the criteria is more reliable.  The program implements the calculation at the end of the training process and every epoch process through two versions independently on keras.

这个仓库的程序是基于keras计算Precision, recall, f1-score, AUC, loss, accuracy值以及绘制ROC曲线。keras自带的Precision, recall, f1-score是基于
一个batch图片的结果，这样显然是不能描述整个训练数据集或者验证集的metrics。我在python库sklearn上写了一个程序基于整个数据集（one epoch)计算上述metrics。

本仓库的程序很简单，用一个程序文件即可实现功能。程序是通过跑一个具体的深度学习模型展示其计算过程。
四个程序中相同文件名的后缀.ipynb文件和.py文件是一样的程序，只是文件形式不同，.ipynb用jupyter打开，.py则是用spyder打开，建议使用.ipynb打开程序。
criteria_end_of_epoch是每一个epoch后计算一次训练集和验证集的metrics，且在所有epoch训练完成后绘制metrics-epoch图以观察训练过程中metrics的变化情况。
criteria_end_of_train是在训练完成后计算训练集和验证集的metrics，因为对于大型的图片数据集，如果每一个epoch都计算metrics，那么将消耗很多计算资源并且训练时间会很长。

程序需要设置的参数，比如：训练集目录、验证集目录等等，直接在程序文件内操作。绝大多数参数是在.ipynb的第二个单元格中修改，其他修改参数可看注释，注释已明确说明。
另外因为metrics计算是基于具体深度学习模型的，我简单的搭建一个模型作为示范，你们应用本metrics程序到自己模型上时，可以通过load_model的方式后者直接在设计模型的单元格内敲模型设计的代码。

