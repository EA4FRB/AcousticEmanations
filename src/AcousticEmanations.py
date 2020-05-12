import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QFileDialog
from PyQt5.QtGui import QIcon

from PyQt5.QtCore import QSize, QCoreApplication, QSettings

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import librosa
import librosa.display

import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import os
import pyaudio
import wave
from time import sleep
from MainWindow import Ui_MainWindow

import shutil

import numpy as np

ORGANIZATION_NAME = 'Melchor Varela - EA4FRB'
ORGANIZATION_DOMAIN = 'sark110.com'
APPLICATION_NAME = 'Acoustic Emanations Tool'
COPYRIGHT_DATE = "© 2020,"

class MainWindow(QMainWindow, Ui_MainWindow):
    _trim_threshold = 15

    _key_name = 'key0'
    _key_idx = 0

    _path_train = "../train"
    _path_models = "./saved_models/"
    _metadata_name = "train_data.csv"

    _train_data = []

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.restore_settings()

        self.progressBarRecord.setValue(0)
        self.progressBarCompute.setValue(0)

        self.pushButtonRec.pressed.connect(self.button_record_slot)
        self.pushButtonLoad.pressed.connect(self.button_load_slot)

        self.lineEditKeyVal.editingFinished.connect(self.edit_key_val_slot)
        self.lineEditKeyVal.setText(self._key_name)

        self.spinBoxRecNb.setValue(self._key_idx)
        self.spinBoxRecNb.valueChanged.connect(self.spinbox_rec_nb_slot)

        self.pushButtonModel.pressed.connect(self.button_model_slot)

        self.labelFileName.setText(self._key_name + '-' + str(self._key_idx) + '.wav')

        self.actionAbout.triggered.connect(self.about)
        self.actionFolderSelect.triggered.connect(self.folder_select)

        os.makedirs(self._path_train + '/wav/', exist_ok=True)
        os.makedirs(self._path_train + '/metadata/', exist_ok=True)
        os.makedirs(self._path_models, exist_ok=True)

        try:
            self._train_data = pd.read_csv(self._path_train + '/metadata/' + self._metadata_name)
        except:
            self._train_data = []

        self.statusBar().showMessage(self._path_train)
        self.show()

    def button_model_slot(self):
        self.clear_model_indicators()

        self.progressBarCompute.setValue(10)
        featuresdf = self.compile_features()
        if len(featuresdf) == 0:
            QMessageBox.about(self, "Error", "No training data")
            return

        self.labelFeatures.setText('File:' + self._path_train + '/metadata/' + self._metadata_name + ', length: ' + str(len(featuresdf)))

        from sklearn.preprocessing import LabelEncoder
        from keras.utils import to_categorical

        import numpy as np

        # Convert features and corresponding classification labels into numpy arrays
        X = np.array(featuresdf.feature.tolist())
        y = np.array(featuresdf.class_label.tolist())

        # Encode the classification labels
        le = LabelEncoder()
        yy = to_categorical(le.fit_transform(y))

        # split the dataset
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)

        import numpy as np
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten
        from keras.layers import Convolution2D, MaxPooling2D
        from keras.optimizers import Adam
        from keras.utils import np_utils
        from sklearn import metrics

        num_labels = yy.shape[1]
        filter_size = 2

        # Construct model
        model = Sequential()

        model.add(Dense(256, input_shape=(40,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_labels))
        model.add(Activation('softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        # Display model architecture summary
        model.summary()

        # Calculate pre-training accuracy
        score = model.evaluate(x_test, y_test, verbose=0)
        accuracy = 100 * score[1]

        print("Pre-training accuracy: %.4f%%" % accuracy)
        self.labelPreTrainAccuracy.setText("Pre-training accuracy: %.4f%%" % accuracy)

        from keras.callbacks import ModelCheckpoint
        from datetime import datetime

        self.progressBarCompute.setValue(30)

        num_epochs = 100
        num_batch_size = 32

        checkpointer = ModelCheckpoint(filepath=self._path_models + 'weights.best.basic_mlp.hdf5',
                                       verbose=1, save_best_only=True)
        start = datetime.now()

        self.progressBarCompute.setValue(40)

        model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),
                  callbacks=[checkpointer], verbose=1)

        duration = datetime.now() - start
        print("Training completed in time: ", duration)

        # Evaluating the model on the training and testing set
        score = model.evaluate(x_train, y_train, verbose=0)
        print("Training Accuracy: ", score[1])
        self.labelTrainAccuracy.setText("Training Accuracy: " + str(score[1]))

        score = model.evaluate(x_test, y_test, verbose=0)
        print("Testing Accuracy: ", score[1])
        self.labelTestAccuracy.setText("Testing Accuracy: " + str(score[1]))

        from sklearn.metrics import confusion_matrix

        snn_pred = model.predict(x_test, batch_size=32, verbose=0)
        snn_predicted = np.argmax(snn_pred, axis=1)
        snn_cm = confusion_matrix(np.argmax(y_test, axis=1), snn_predicted)

        self.progressBarCompute.setValue(80)

        self.plot_confusion_matrix(snn_cm, normalize=True, target_names=self._metadata.class_name.unique(),
                                   title="Confusion Matrix")
        self.progressBarCompute.setValue(100)
        
    def extract_features(self, file_name):
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
            audio_n = librosa.util.normalize(audio)
            audio_trim, index = librosa.effects.trim(audio_n, top_db=self._trim_threshold)
            mfccs = librosa.feature.mfcc(y=audio_trim, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T, axis=0)

        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None

        return mfccsscaled

    def compile_features(self):
        try:
            self._metadata = pd.read_csv(self._path_train + '/metadata/' + self._metadata_name)
            features = []
            # Iterate through each sound file and extract the features
            for index, row in self._metadata.iterrows():
                file_name = os.path.join(self._path_train + '/wav/' + str(row["slice_file_name"]))
                class_label = row["class_name"]
                data = self.extract_features(file_name)
                features.append([data, class_label])

            # Convert into a Panda dataframe
            return pd.DataFrame(features, columns=['feature', 'class_label'])
        except:
            return []

    def plot_confusion_matrix(self, cm,
                              target_names,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        self.MplWidgetMatrix.canvas.axes.clear()
        # self.MplWidgetMatrix.canvas.axes.figure(figsize=(8, 6))
        self.MplWidgetMatrix.canvas.axes.imshow(cm, interpolation='nearest', cmap=cmap)
        self.MplWidgetMatrix.canvas.axes.set_title(title)
        #        self.MplWidgetMatrix.canvas.axes.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            self.MplWidgetMatrix.canvas.axes.set_xticks(tick_marks)
            self.MplWidgetMatrix.canvas.axes.set_xticklabels(target_names)
            self.MplWidgetMatrix.canvas.axes.set_yticks(tick_marks)
            self.MplWidgetMatrix.canvas.axes.set_yticklabels(target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                self.MplWidgetMatrix.canvas.axes.text(j, i, "{:0.2f}".format(cm[i, j]),
                                                      horizontalalignment="center",
                                                      color="white" if cm[i, j] > thresh else "black")
            else:
                self.MplWidgetMatrix.canvas.axes.text(j, i, "{:,}".format(cm[i, j]),
                                                      horizontalalignment="center",
                                                      color="white" if cm[i, j] > thresh else "black")

        # self.MplWidgetMatrix.canvas.axes.tight_layout()
        self.MplWidgetMatrix.canvas.axes.set_ylabel('True label')
        self.MplWidgetMatrix.canvas.axes.set_xlabel(
            'Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
        self.MplWidgetMatrix.canvas.draw()

    def edit_key_val_slot(self):
        self._key_name = self.lineEditKeyVal.text()
        self._key_idx = 0
        self.labelFileName.setText(self._key_name + '-' + str(self._key_idx) + '.wav')
        self.spinBoxRecNb.setValue(self._key_idx)
        self.save_settings()

    def spinbox_rec_nb_slot(self, value):
        self._key_idx = value
        self.labelFileName.setText(self._key_name + '-' + str(self._key_idx) + '.wav')
        self.save_settings()

    def clear_model_indicators(self):
        self.labelTrainAccuracy.setText('')
        self.labelTestAccuracy.setText('')
        self.labelPreTrainAccuracy.setText('')

    def console_output(self, msg):
        # Fetch text already in QTextBrowser
        pre_text = self.textbrowser_console.toPlainText()
        # Newline addition only if pre_text exist
        if pre_text:
            self.textBrowserConsole.setText(self.qtranslate(self.centralwidget_name, (pre_text + '\n' + str(msg))))
        else:
            self.textBrowserConsole.setText(self.qtranslate(self.centralwidget_name, str(msg)))
        # Move cursor view to the end - scroll view
        self.textBrowserConsole.moveCursor(QtGui.QTextCursor.End)
        # Repaint to force event loop to render at each call
        self.textBrowserConsole.repaint()

    def button_record_slot(self):
        filename = self._key_name + '-' + str(self._key_idx) + '.wav'
        tmpfile = "temp.wav"
        self.rec_file(tmpfile)
        self._train_data.append([filename, self._key_name])
        shutil.copyfile(tmpfile, self._path_train + '/wav/' + filename)
        os.remove(tmpfile)
        self.plot_record(filename, self._path_train + '/wav/' + filename)

        self._key_idx = self._key_idx + 1
        self.spinBoxRecNb.setValue(self._key_idx)
        self.labelFileName.setText(self._key_name + '-' + str(self._key_idx) + '.wav')

        df = pd.DataFrame(self._train_data, columns=['slice_file_name', 'class_name'])
        df.to_csv(self._path_train + '/metadata/' + self._metadata_name, index=False)

        self.save_settings()

    def button_load_slot(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Open wav file", self._path_train + '/wav/',
                                                  "All Files (*);;Wave Files (*.wav)", options=options)
        if filename:
            self.plot_record(os.path.basename(filename), filename)

    def plot_record(self, name, filename):
        audio, sample_rate = librosa.load(filename)
        audio_n = librosa.util.normalize(audio)
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(audio_n)
        self.MplWidget.canvas.axes.set_title(name)
        self.MplWidget.canvas.draw()

    def rec_file(self, filename):
        sleep(0.5)
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt24
        channels = 1
        fs = 44100  # Record at 44100 samples per second
        seconds = 3

        p = pyaudio.PyAudio()  # Create an interface to PortAudio
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []  # Initialize array to store frames

        # Store data in chunks for 3 seconds
        max = int(fs / chunk * seconds)
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)
            self.progressBarRecord.setValue(i * 100 / (max - 1))

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()
        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    def folder_select(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory", self._path_train))
        if folder:
            self._path_train = folder
            self.save_settings()
            try:
                self._train_data = pd.read_csv(self._path_train + '/metadata/' + self._metadata_name)
            except:
                self._train_data = []
        self.statusBar().showMessage(self._path_train)

    def about(self):
        QMessageBox.about(self, "About", APPLICATION_NAME + '\n\n' + COPYRIGHT_DATE + ' ' + ORGANIZATION_NAME)

    def restore_settings(self):
        settings = QSettings()
        self._path_train = settings.value('settings/path_train', defaultValue=self._path_train, type=str)
        self._key_name = settings.value('settings/key_name', defaultValue=self._key_name, type=str)
        self._key_idx = settings.value('settings/key_index', defaultValue=self._key_idx, type=int)

    def save_settings(self):
        settings = QSettings()
        settings.setValue('settings/path_train', self._path_train)
        settings.setValue('settings/key_name', self._key_name)
        settings.setValue('settings/key_index', self._key_idx)
        settings.sync()

if __name__ == '__main__':
    QCoreApplication.setApplicationName(ORGANIZATION_NAME)
    QCoreApplication.setOrganizationDomain(ORGANIZATION_DOMAIN)
    QCoreApplication.setApplicationName(APPLICATION_NAME)

    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
