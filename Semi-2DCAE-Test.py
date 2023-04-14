
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

save_path_result = '../result'

def plot_matrix(y_true, y_pred,title_name):
    cm = confusion_matrix(y_true, y_pred)
    ax = sn.heatmap(cm,cmap='OrRd',vmin=0, vmax=100,fmt='g',xticklabels=['chat', 'email', 'file','p2p','streaing','voip'],yticklabels=['chat', 'email', 'file','p2p','streaing','voip'])
    ax.set_title(title_name)
    ax.set_xlabel('predict') 
    ax.set_ylabel('true') 

def mul(Spectrum, sample):
    similar=[]
    for value in sample:
        idx=np.abs(Spectrum - np.asarray(value)).argmin()
        similar.append(Spectrum[idx])
    P=np.exp(-np.abs(np.asarray(sample)-np.asarray(similar))).mean()
    return P
def softmax(data):
    exp=np.exp(np.asarray(list(data.values()))) 
    return dict(zip(data.keys(),exp/exp.sum()))

class FlowSpectrum_detect:
    def __init__(self,X_test,y_test,test_Label,reconstructor):
        self.test_Label=test_Label
        self.X_test=X_test       
        self.y_test=y_test 
        self.reconstructor=reconstructor
        self.X_test_decompose=self.reconstructor.predict(X_test).reshape(-1)

    def detect(self,sample):
        FlowSpectrum = np.load('../result/FlowSpectrum.npy', allow_pickle=True).item()
        result={}
        for label in FlowSpectrum.keys():
            result[label] = mul(Spectrum=FlowSpectrum[label], sample=sample)
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
    ModelName = 'Semi-2DCAE'
    model = load_model('../result/Semi-2DCAE_model.h5')
    Encoder = Model(inputs=[model.input], outputs=[model.get_layer('z2').output], name='Encoder')

    Labels = [0.0,1.0,2.0,3.0,4.0,5.0]
    x_test=dataprocessing.decode_idx3_ubyte("../Vpn/images")
    y_test1=dataprocessing.decode_idx1_ubyte("../Vpn/labels")
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