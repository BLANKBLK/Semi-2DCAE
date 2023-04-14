
#%%
import dataprocessing
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,Input,Concatenate,Reshape,UpSampling2D,PReLU
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam,SGD,RMSprop,Adagrad,Adadelta,Adamax
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import seaborn as sns

x_train=dataprocessing.decode_idx3_ubyte("../Vpn/train-images")
y_train1=dataprocessing.decode_idx1_ubyte("../Vpn/train-labels")

x_train, x_verification, y_train1, y_verification1 = train_test_split(x_train, y_train1, test_size=0.33, random_state=42)

x_train = x_train.reshape(-1, 28, 28, 1)/255.0
x_verification = x_verification.reshape(-1, 28, 28, 1)/255.0
y_train = to_categorical(y_train1, num_classes=6)
y_verification = to_categorical(y_verification1, num_classes=6)
activation=PReLU
adam = Adam(lr=0.0005)
input=Input(shape=(28, 28, 1),name='input')
encoder = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", kernel_regularizer=regularizers.L2(0.01))(input)
encoder = PReLU()(encoder)
encoder = MaxPooling2D(pool_size=2, strides=2, padding='same')(encoder)
encoder = Conv2D(filters=32, kernel_size=3, strides=1, padding="same", kernel_regularizer=regularizers.L2(0.01))(encoder)
encoder = PReLU()(encoder)
encoder = MaxPooling2D(pool_size=2, strides=2, padding='same')(encoder)
encoder = Flatten()(encoder)

F=Dense(units=1,name='F')(encoder)
T=Dense(units=1,name='T')(encoder)
encoder=Concatenate(name='code')([F,T])
decoder = Dense(1568, activation='relu')(encoder)
decoder = Reshape((7,7,32))(decoder)
decoder = Conv2D(filters=32, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_uniform(seed=None))(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_uniform(seed=None))(decoder)
decoder = UpSampling2D((2, 2))(decoder)
output  = Conv2D(1, (3, 3), activation='sigmoid', padding='same',name='output')(decoder)
result=Dense(units=6,activation='softmax',name='result')(T)
                                                               
AutoEncoder=Model(inputs=[input], outputs=[output], name='AutoEncoder')
Encoder=Model(inputs=[input], outputs=[encoder], name='Encoder')

model=Model(inputs=[input],outputs=[output,result],name='Model')
model.compile(
    optimizer=adam,
    loss={
        
        'output': 'mse',
        'result': 'categorical_crossentropy'
    },
    loss_weights={
        'output': 1.,
        'result': 1.
    },
    metrics={
        'result': 'accuracy'
    }
)

#%%
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
save_path_result = "../result"
model.summary()

plot_model(model, to_file=save_path_result+'/'+'Semi-2DCAE_model.png', show_shapes=True, )
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint=ModelCheckpoint(
    filepath=save_path_result +'/'+'Semi-2DCAE_model.h5',
    monitor='val_output_loss',verbose=1,
    save_best_only=True,mode='min'
)

history=model.fit(
    {'input':x_train},{'output':x_train,'result':y_train},
    validation_data=({'input':x_verification},{'output':x_verification,'result':y_verification}),
    batch_size=64, epochs=300,
    callbacks=[checkpoint]
)

plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['output_loss'],label='output_loss')
plt.plot(history.history['val_output_loss'],label='val_output_loss')
plt.plot(history.history['result_loss'],label='result_loss')
plt.plot(history.history['val_result_loss'],label='val_result_loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(save_path_result +'/'+'loss.png')
plt.plot(history.history['result_accuracy'],label='result_acc')
plt.plot(history.history['val_result_accuracy'],label='val_result_acc')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(save_path_result +'/'+'accuracy.png')
#%%
import numpy as np
class FlowSpectrum:
    def __init__(self,X,y,Labels,reconstructor,Code):
        self.Labels=Labels
        self.X=X
        self.reconstructor=reconstructor
        self.Code=Code
        self.X_decompose_=self.reconstructor.predict(X)
        X_decompose=self.X_decompose_.reshape(-1)
        self.X_lim=[X_decompose.min(),X_decompose.max()]
        self.FlowSpectrum={}
        for label in Labels:
            self.FlowSpectrum[label]=X_decompose[y==label]
        np.save(save_path_result+'/'+'FlowSpectrum.npy', self.FlowSpectrum) 
    def get_FlowSpectrum(self,title,save_path=save_path_result+'/'+'FlowSpectrum.png'):
        plt.figure(figsize=(15,len(self.Labels)),dpi=400)
        for i,label in enumerate(self.Labels):
            plt.subplot(len(Labels),1,i+1)
            sns.rugplot(self.FlowSpectrum[label], height=1.,c='r')
            plt.xlim(self.X_lim)
            plt.title(class6_vpn[label])
        plt.suptitle(title)
        plt.tight_layout()
        if save_path==None:
            plt.show()
        else:
            plt.savefig(save_path)
    def show_latent_representation(self):
        latent_representation = self.Code.predict(self.X)
        plt.figure(figsize=(10, 10))
        x=[x[0] for x in latent_representation]
        y=[y[1] for y in latent_representation]
        z =  latent_representation[:,0].reshape(-1)
        w = latent_representation[:,1].reshape(-1)
        # x=[x for x in z]
        # y=[y for y in w]
        ax=sns.scatterplot(
                        z,    
                        w,                       
                        hue=y_train1,
                        markers=markers,
                        palette='tab10')
        plt.xlabel("Features Represent F")
        plt.ylabel("Features Represent T")
        plt.legend(bbox_to_anchor=(1, 1),loc=2,borderaxespad=0)
        plt.savefig(save_path_result +'/'+'2D.png')
                
ModelName = 'Semi-2DCAE'
model = load_model('../result/Semi-2DCAE_model.h5')
Encoder = Model(inputs=[model.input], outputs=[model.get_layer('T').output], name='Encoder')
coder = Model(inputs=[model.input], outputs=[model.get_layer('code').output], name='coder')
class6_vpn = {0.0:'Chat',1.0:'Email',2.0:'File',3.0:'P2p',4.0:'Streaming',5.0:'Voip'}
markers = {"Chat": "0.0", "Email": "1.0",'File':"2.0",'P2p':"3.0",'Streaming':"4.0",'Voip':"5.0"}
Labels = [0.0,1.0,2.0,3.0,4.0,5.0]
FS=FlowSpectrum(
    X=x_train,y=y_train1,
    Labels=Labels,
    reconstructor=Encoder,
    Code = coder,  
)
FS.get_FlowSpectrum(title=f'FlowSpectrum Based on {ModelName}')
FS.show_latent_representation()