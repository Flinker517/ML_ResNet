from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.optimizers import SGD
import numpy as np
import model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os
import tensorflow.keras.backend as K
import math

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#tf.config.gpu.set_per_process_memory_fraction(0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

os.environ['CUDA_VISIBLE_DEVICES']='1'
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data(dataset, mode):
    print("Load data...")
    data = np.load(dataset)
    # print(data.files)
    img = None
    label = None
    if mode == "train":
        img = data['train_images']
        label = data['train_labels']
    if mode == "test":
        img = data['test_images']
        label = data['test_labels']
    if mode == "val":
        img = data['val_images']
        label = data['val_labels']
    return img, label


trainimg, trainlabel = load_data("/home/gtyan/dataset/hw1/data/chestmnist.npz", "train")
valimg, vallabel = load_data("/home/gtyan/dataset/hw1/data/chestmnist.npz", "val")
testimg, testlabel = load_data("/home/gtyan/dataset/hw1/data/chestmnist.npz", "test")
trainimg = trainimg.reshape((trainimg.shape[0], 28, 28, 1))
valimg = valimg.reshape((valimg.shape[0], 28, 28, 1))
testimg = testimg.reshape((testimg.shape[0], 28, 28, 1))

trainimg = trainimg.astype("float32") / 255.0
testimg = testimg.astype("float32") / 255.0
valimg = valimg.astype("float32") / 255.0
trainimg = (trainimg - 0.5)/0.5
testimg = (testimg - 0.5)/0.5
valimg = (valimg - 0.5)/0.5

trainlabel = trainlabel.astype("float32")
vallabel = vallabel.astype("float32")
testlabel = testlabel.astype("float32")
'''le = MultiLabelBinarizer()
trainlabel = le.fit_transform(trainlabel)
testlabel = le.transform(testlabel)
vallabel = le.transform(vallabel)'''

optimizer = SGD(lr=0.001,momentum=0.9)
model1 = model.create_model_resnet50((28, 28, 1), 14)

'''def myBCE(y_true,y_pred):
   # max_val = np.clip(-y_true, min=0)
    loss=0
    for i in range(0,len(y_true)):
        loss+= y_true[i]*math.log(y_pred[i]) + (1-y_true[i])*math.log(1-y_pred)[i]
    return loss'''
'''def myBCE(y_true, y_pred):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
    y_pred=K.print_tensor(y_pred)
    return tf.reduce_mean(loss(y_true, y_pred)) #tf.reduce_mean(loss(y_true, y_pred))'''
'''def myBCE(y_true, y_pred):
    y_pred=K.print_tensor(y_pred)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)'''
'''def myBCE(y_true, y_pred):
    y_pred=K.print_tensor(y_pred)
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)'''

#loss = tf.reduce_mean(xent(targets, pred) * weights)
#model1.compile(optimizer=optimizer, loss=myBCE, metrics=['accuracy'])
model1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = EarlyStopping(monitor='val_accuracy',verbose=1,mode='max',patience=5)
h = model1.fit(trainimg, trainlabel, batch_size=128, epochs=40, validation_data=(valimg, vallabel))#,callbacks=[checkpoint])
# model.save('drive/My Drive/Colab Notebooks/path7.h5')
acc, loss = model1.evaluate(x=testimg, y=testlabel, verbose=1)
acc1 = h.history['accuracy']
loss1 = h.history['loss']
acc2 = h.history['val_accuracy']
loss2 = h.history['val_loss']
np_acc1 = np.array(acc1).reshape((1,len(acc1)))
np_loss1 = np.array(loss1).reshape((1,len(loss1)))
np_acc2 = np.array(acc2).reshape((1,len(acc2)))
np_loss2 = np.array(loss2).reshape((1,len(loss2)))
np_out = np.concatenate([np_acc1,np_loss1,np_acc2,np_loss2],axis=0)
np.savetxt('/home/gtyan/dataset/hw1/output/chest_train.txt',np_out)    
print("保存训练结果成功")

data = open("/home/gtyan/dataset/hw1/output/chest_test.txt", 'w')
data.write(str(acc)+' '+str(loss))
data.close()
print("保存测试结果成功") 
