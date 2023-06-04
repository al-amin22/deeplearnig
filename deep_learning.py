import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()

kategori = ['T-shirt/top',
            'Touser',
            'pullover',
            'Dress',
            'coat',
            'sandal',
            'shirt',
            'sneakers',
            'bag',
            'ankle-boot']
#menampilkan 1 gambar dengan acak dengan menggunakan fungsi random
gambar = random.randint(0, len(x_train))
plt.Figure()
plt.imshow(x_train[gambar,:,:], cmap='gray')
plt.title(' nomor gambar secara acak ={} - kategori {}'.format(gambar, int(y_train[gambar])))
plt.show()

#menampilkan gambar dengan ukuran 12x12
nrows = 12
ncols =12
fig, axes = plt.subplots(nrows, ncols)
axes = axes.ravel()
ntraining = len(x_train)
for i in np.arange(0, nrows*ncols):
    indexku = random.randint(0, ntraining)
    axes[i].imshow(x_train[indexku], cmap='gray')
    axes[i].set_title(int(y_train[indexku]), fontsize=8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)

#menormalisasikan dataset
x_train = x_train/255
x_test = x_test/255

#membagi data training menjadi validate data

from sklearn.model_selection import train_test_split
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train,
                                                            test_size=0.2,
                                                            random_state=123)

#mengubah data menjadi 3 dimensi

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_validate = x_validate.reshape(x_validate.shape[0], 28,28,1)

#mengimpor library keras untuk melkukan training yaitun conv2d

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam

classifier = Sequential()
classifier.add(Conv2D(32, [3,3],  input_shape=(28,28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))


#membuat flattenign dan model CN NN neural network
classifier.add(Flatten())
classifier.add(Dense(activation='relu', units=32))
classifier.add(Dense(activation='sigmoid', units=10))
classifier.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), 
                   metrics=['accuracy'])
classifier.summary()
#mmebuat visuallisasi dari CNN yang telah dibuat sebelumnya
from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file='model_CNN_ku.png',
           show_layer_names=True,
           show_shapes= True)

#mulai melakukan training data
run_model = classifier.fit(x_train, y_train, batch_size=500, verbose=1, nb_epoch=30, 
                           validation_data=(x_validate, y_validate))

print(run_model.history.keys())

plt.plot(run_model.history['accuracy'])
plt.plot(run_model.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()



plt.plot(run_model.history['loss'])
plt.plot(run_model.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()


#mengevaluasi model CNN kita

evaluasi = classifier.evaluate(x_test, y_test)
print('test accuracy = {:.2f}%'.format(evaluasi[1]*100 ))


classifier.save('model_cnn.hd5', include_optimizer=True)
print('model telah disimpan')

#MEMPREDIKSI kategori di tesy set

hasil_prediksi=classifier.predict_classes(x_test)

#membuat plot hasil prediksi
fig, axes = plt.subplots(5, 5)
axes = axes.ravel()
for i in np.arange(0, 5*5):
    axes[i].imshow(x_test[i]. reshape(28,28), cmap='gray')
    axes[i].set_title('hasil prediksi = {}\n label asli {}'.format(hasil_prediksi[i], y_test[i]))
    axes[i].axis('off')
from sklearn.metrics import confusion_matrix
import pandas as pd
cm = confusion_matrix(y_test, hasil_prediksi)
cm_label = pd.DataFrame(cm, columns = np.unique(y_test), index = np.unique(y_test))
cm_label.index.name = 'asli'
cm_label.columns.name = 'prediksi'
plt.Figure(figsize=(14,10))
sns.heatmap(cm_label, annot=True)


from sklearn.metrics import classification_report
jumlah_kategori = 10
nama_target= ['kategori {} '.format(i) for i in range(jumlah_kategori)]
print(classification_report(y_test, hasil_prediksi, target_names=nama_target))
