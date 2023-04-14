
#%%
import dataprocessing
from sklearn.decomposition import PCA,TruncatedSVD,FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from datetime import datetime
import tensorflow as tf
import itertools

save_path_result = '/root/0-Semi-2DCAE-idx-VpnSessionAllLayers/result'

def plot_matrix(y_true, y_pred,title_name):
    cm = confusion_matrix(y_true, y_pred)#混淆矩阵
    #annot = True 格上显示数字 ，fmt：显示数字的格式控制
    ax = sn.heatmap(cm,cmap='OrRd',vmin=0, vmax=100,fmt='g',xticklabels=['chat', 'email', 'file','p2p','streaing','voip'],yticklabels=['chat', 'email', 'file','p2p','streaing','voip'])
    #xticklabels、yticklabels指定横纵轴标签
    ax.set_title(title_name) #标题s
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴

def mul(Spectrum, sample):
    similar=[]
    for value in sample:
        idx=np.abs(Spectrum - np.asarray(value)).argmin()#abs绝对值函数；argmin()表示使目标函数f(x)取最小值时的变量值
        similar.append(Spectrum[idx])
    P=np.exp(-np.abs(np.asarray(sample)-np.asarray(similar))).mean()#mean()求取均值 经常操作的参数为axis,以m * n矩阵举例:axis 不设置值
    return P
def softmax(data):
    exp=np.exp(np.asarray(list(data.values()))) #values()返回一个字典中的值
    return dict(zip(data.keys(),exp/exp.sum()))

class FlowSpectrum_detect:
    def __init__(self,X_test,y_test,test_Label,reconstructor):
        self.test_Label=test_Label
        self.X_test=X_test       
        self.y_test=y_test 
        self.reconstructor=reconstructor
        self.X_test_decompose=self.reconstructor.predict(X_test).reshape(-1)

    def detect(self,sample):
        FlowSpectrum = np.load('/root/0-Semi-2DCAE-idx-VpnSessionAllLayers/result/FlowSpectrum.npy', allow_pickle=True).item()
        result={}
        for label in FlowSpectrum.keys():
            result[label] = mul(Spectrum=FlowSpectrum[label], sample=sample)
        # 可以增加softmax层
        return softmax(result)

    def detection_test(self,outputdict=None,name=None,sample_count=100,sample_size=10):
        y_pred,y_true = [],[]
        times=[]
        for label in self.test_Label:
            data = self.X_test_decompose[self.y_test==label]
            for i in range(sample_count):
                sample = np.random.choice(a=data, size=sample_size)
                start=datetime.now()
                result = self.detect(sample=sample)
                end=datetime.now()
                y_pred.append(max(result, key=lambda k: result[k]))
                y_true.append(label)
                times.append((end-start).total_seconds())
        np.set_printoptions(precision=2)
        plt.figure()
        plot_matrix(
            y_true, y_pred, title_name='Semi-2DCAE-Vpn-confusion_matrix')
        plt.savefig(save_path_result+'/'+'confusion_matrix.png')
        print('time:',np.mean(times))
        print(accuracy_score(y_pred=y_pred, y_true=y_true))
        print(classification_report(y_pred=y_pred, y_true=y_true,digits=4))
        if outputdict != None and name != None:
            pd.DataFrame(classification_report(
                y_pred=y_pred, y_true=y_true, digits=6, output_dict=True
            )).transpose().to_csv(save_path_result+'/'+'_Report.csv')
            pd.DataFrame(confusion_matrix(
                y_pred=y_pred, y_true=y_true, labels=self.test_Label
            ), columns=self.test_Label, index=self.test_Label).to_csv(save_path_result+'/'+'_Matrix.csv')


from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
if __name__=='__main__':
    ModelName = 'Smi-2DCAE'
    model = load_model('/root/0-Semi-2DCAE-idx-VpnSessionAllLayers/result/Semi-2DCAE_model.h5')
    Encoder = Model(inputs=[model.input], outputs=[model.get_layer('z2').output], name='Encoder')

    Labels = [0.0,1.0,2.0,3.0,4.0,5.0]
    
    x_test=dataprocessing.decode_idx3_ubyte("/root/0-Semi-2DCAE-idx-VpnSessionAllLayers/VpnSessionAllLayers/t10k-images-idx3-ubyte")
    y_test1=dataprocessing.decode_idx1_ubyte("/root/0-Semi-2DCAE-idx-VpnSessionAllLayers/VpnSessionAllLayers/t10k-labels-idx1-ubyte")
    x_test = x_test.reshape(-1, 28, 28, 1)/255.0
    
    FS=FlowSpectrum_detect(
        X_test=x_test, y_test=y_test1,
        test_Label=Labels,
        reconstructor=Encoder,
    )
    FS.detection_test(
        sample_count=1000, sample_size=10,
        outputdict='Result', name=ModelName
    )