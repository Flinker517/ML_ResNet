# ML_ResNet
这是我们组的MedMNIST十项全能小作业代码。\\
data文件夹下放置MedMNIST的十个数据集，由于github限制，此处没有数据集。
code文件夹下放置我们的代码，其中model.py和resnet.py是ResNet50的Keras实现：model.py是ResNet50的Keras版本，resnet.py是运行文件，由于某个误差函数（BCEWitLogitsLoss)的差异，最终改用PyTorch。
其余train_for_xx.py即为相应数据集的训练代码，命令行python运行即可，它会输出测试结果，同时保存模型到output1文件夹下（需要提前建立）。res50.py是ResNet50的PyTorch版本。
