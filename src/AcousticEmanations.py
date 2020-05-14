# ---------------------------------------------------------
"""
  This file is a part of the "Acoustic Emanations Tool" software

  MIT License

  @author Copyright (c) 2020 Melchor Varela - EA4FRB

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""
# ---------------------------------------------------------

import itertools
import os
import shutil
import sys
import wave
from datetime import datetime
from time import sleep

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaudio
from MainWindow import Ui_MainWindow
from PyQt5.QtCore import QCoreApplication, QSettings
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, scale
from commondefs import *


class MainWindow(QMainWindow, Ui_MainWindow):
    _key_name = DEF_KEY_NAME
    _key_idx = 0

    _path_train = PATH_TRAIN_DEF
    _metadata_name = METADATA_FILENAME

    _train_data = []

    _model = []

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.restore_settings()

        self.progressBarRecord.setValue(0)
        self.progressBarCompute.setValue(0)

        self.pushButtonRec.pressed.connect(self.button_record_action)

        self.pushButtonTest.pressed.connect(self.button_test_action)

        self.lineEditKeyVal.editingFinished.connect(self.edit_key_val_action)
        self.lineEditKeyVal.setText(self._key_name)

        self.spinBoxRecNb.setValue(self._key_idx)
        self.spinBoxRecNb.valueChanged.connect(self.spinbox_rec_nb_action)

        self.pushButtonModel.pressed.connect(self.model_process_action)

        self.labelFileName.setText(self._key_name + '-' + str(self._key_idx) + '.wav')

        self.actionAbout.triggered.connect(self.about)
        self.actionFolderSelect.triggered.connect(self.folder_select_action)
        self.actionLoadWaveFile.triggered.connect(self.load_wave_file_action)

        self.clear_model_indicators()

        os.makedirs(self._path_train + SUBPATH_WAV, exist_ok=True)
        os.makedirs(self._path_train + SUBPATH_META, exist_ok=True)
        os.makedirs(self._path_train + SUBPATH_MODELS, exist_ok=True)

        try:
            self._train_data = pd.read_csv(self._path_train + SUBPATH_META + self._metadata_name)
        except Exception as e:
            self._train_data = []

        self.statusBar().showMessage(self._path_train)
        self.show()

    def model_process_action(self):
        self.clear_model_indicators()

        self.progressBarCompute.setValue(10)
        featuresdf = self.compile_features()
        if len(featuresdf) == 0:
            QMessageBox.about(self, "Error", "No training data")
            return

        featuresdf.to_csv(self._path_train + SUBPATH_MODELS + FEATURES_FILENAME, index=False)

        self.labelFeatures.setText(
            'File:' + self._path_train + SUBPATH_META + self._metadata_name + ', length: ' + str(len(featuresdf)))

        # Convert features and corresponding classification labels into numpy arrays
        X = np.array(featuresdf.feature.tolist())
        y = np.array(featuresdf.class_label.tolist())

        # Encode the classification labels
        self._le = LabelEncoder()
        self._yy = to_categorical(self._le.fit_transform(y))

        # split the dataset
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(X, self._yy, test_size=0.2,
                                                                                    random_state=42)

        num_labels = self._yy.shape[1]
        filter_size = 2

        # Construct model
        self._model = Sequential()

        self._model.add(Dense(256, input_shape=(40,)))
        self._model.add(Activation('relu'))
        self._model.add(Dropout(0.5))

        self._model.add(Dense(256))
        self._model.add(Activation('relu'))
        self._model.add(Dropout(0.5))

        self._model.add(Dense(num_labels))
        self._model.add(Activation('softmax'))

        # Compile the model
        self._model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        # Display model architecture summary
        self._model.summary()

        # Calculate pre-training accuracy
        score = self._model.evaluate(self._x_test, self._y_test, verbose=0)
        accuracy = 100 * score[1]

        print("Pre-training accuracy: %.4f%%" % accuracy)
        self.labelPreTrainAccuracy.setText("Pre-training accuracy: %.4f%%" % accuracy)

        self.progressBarCompute.setValue(30)

        num_epochs = 100
        num_batch_size = 32

        checkpointer = ModelCheckpoint(filepath=self._path_train + SUBPATH_MODELS + CHECKPOINT_FILENAME,
                                       verbose=1, save_best_only=True)
        start = datetime.now()

        self.progressBarCompute.setValue(40)

        self._model.fit(self._x_train, self._y_train, batch_size=num_batch_size, epochs=num_epochs,
                        validation_data=(self._x_test, self._y_test),
                        callbacks=[checkpointer], verbose=1)

        duration = datetime.now() - start
        print("Training completed in time: ", duration)

        # Evaluating the model on the training and testing set
        score = self._model.evaluate(self._x_train, self._y_train, verbose=0)
        print("Training Accuracy: ", score[1])
        self.labelTrainAccuracy.setText("Training Accuracy: " + str(score[1]))

        score = self._model.evaluate(self._x_test, self._y_test, verbose=0)
        print("Testing Accuracy: ", score[1])
        self.labelTestAccuracy.setText("Testing Accuracy: " + str(score[1]))

        snn_pred = self._model.predict(self._x_test, batch_size=32, verbose=0)
        snn_predicted = np.argmax(snn_pred, axis=1)
        snn_cm = confusion_matrix(np.argmax(self._y_test, axis=1), snn_predicted)

        self.progressBarCompute.setValue(80)

        self.plot_confusion_matrix(snn_cm, normalize=True, target_names=self._metadata.class_name.unique(),
                                   title="Confusion Matrix")
        self.progressBarCompute.setValue(100)

    def extract_features(self, file_name):
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
            audio_n = librosa.util.normalize(audio)
            audio_trim, index = librosa.effects.trim(audio_n, top_db=AUD_TRIM_THRESHOLD)
            mfccs = librosa.feature.mfcc(y=audio_trim, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T, axis=0)

        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None

        return mfccsscaled

    def compile_features(self):
        try:
            self._metadata = pd.read_csv(self._path_train + SUBPATH_META + self._metadata_name)
            features = []
            # Iterate through each sound file and extract the features
            for index, row in self._metadata.iterrows():
                file_name = os.path.join(self._path_train + SUBPATH_WAV + str(row["slice_file_name"]))
                class_label = row["class_name"]
                data = self.extract_features(file_name)
                features.append([data, class_label])

            # Convert into a Panda dataframe
            return pd.DataFrame(features, columns=['feature', 'class_label'])
        except Exception as e:
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
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        self.MplWidgetMatrix.canvas.axes.clear()
        # self.MplWidgetMatrix.canvas.axes.figure(figsize=(8, 6))
        self.MplWidgetMatrix.canvas.axes.imshow(cm, origin='lower', interpolation='nearest', cmap=cmap)
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

    def edit_key_val_action(self):
        self._key_name = self.lineEditKeyVal.text()
        self._key_idx = 0
        self.labelFileName.setText(self._key_name + '-' + str(self._key_idx) + '.wav')
        self.spinBoxRecNb.setValue(self._key_idx)
        self.save_settings()

    def spinbox_rec_nb_action(self, value):
        self._key_idx = value
        self.labelFileName.setText(self._key_name + '-' + str(self._key_idx) + '.wav')
        self.save_settings()

    def clear_model_indicators(self):
        self.labelTrainAccuracy.setText('')
        self.labelTestAccuracy.setText('')
        self.labelPreTrainAccuracy.setText('')
        self.labelFeatures.setText('')

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

    def button_record_action(self):
        file_name = self._key_name + '-' + str(self._key_idx) + '.wav'
        tmpfile = "temp.wav"
        self.rec_file(tmpfile)
        self._train_data.append([file_name, self._key_name])
        shutil.copyfile(tmpfile, self._path_train + SUBPATH_WAV + file_name)
        os.remove(tmpfile)
        self.plot_record(file_name, self._path_train + SUBPATH_WAV + file_name)

        self._key_idx = self._key_idx + 1
        self.spinBoxRecNb.setValue(self._key_idx)
        self.labelFileName.setText(self._key_name + '-' + str(self._key_idx) + '.wav')

        df = pd.DataFrame(self._train_data, columns=['slice_file_name', 'class_name'])
        df.to_csv(self._path_train + SUBPATH_META + self._metadata_name, index=False)

        self.save_settings()

    def load_wave_file_action(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open wav file", self._path_train + SUBPATH_WAV,
                                                   "All Files (*);;Wave Files (*.wav)", options=options)
        if file_name:
            self.plot_record(os.path.basename(file_name), file_name)
        self.statusBar().showMessage(self._path_train)

    def plot_record(self, name, file_name):
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        audio_n = librosa.util.normalize(audio)
        audio_trim, index = librosa.effects.trim(audio_n, top_db=AUD_TRIM_THRESHOLD)

        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.set_title(name)
        librosa.display.waveplot(audio_trim, sr=sample_rate, ax=self.MplWidget.canvas.axes)
        self.MplWidget.canvas.draw()

        mfcc = librosa.feature.mfcc(y=audio_trim, sr=sample_rate, n_mfcc=40)
        mfccs = scale(mfcc, axis=0)
        self.MplWidgetSpectrum.canvas.axes.clear()
        self.MplWidgetSpectrum.canvas.axes.set_title(name)
        # librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time', y_axis='mel', ax=self.MplWidgetSpectrum.canvas.axes)
        librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time', y_axis='mel',
                                 ax=self.MplWidgetSpectrum.canvas.axes)
        self.MplWidgetSpectrum.canvas.draw()

    def rec_file(self, file_name):
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
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    def folder_select_action(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory", self._path_train))
        if folder:
            self._path_train = folder
            self.save_settings()
            try:
                self._train_data = pd.read_csv(self._path_train + SUBPATH_META + self._metadata_name)
            except Exception as e:
                self._train_data = []
        self.statusBar().showMessage(self._path_train)

    def button_test_action(self):
        if not self._model:
            QMessageBox.about(self, "Error", "Run model first")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open wav file", self._path_train + SUBPATH_WAV,
                                                   "All Files (*);;Wave Files (*.wav)", options=options)
        if file_name:
            self.plot_record(os.path.basename(file_name), file_name)

            prediction_feature = self.extract_features(file_name)
            if prediction_feature is None:
                QMessageBox.about(self, "Error", "Parsing file: " + file_name)
                return
            predicted_vector = self._model.predict_classes(np.array([prediction_feature]))
            predicted_class = self._le.inverse_transform(predicted_vector)
            outstr = "File: " + os.path.basename(file_name) + '\n'
            outstr += "The predicted class is: " + str(predicted_class[0]) + '\n\n';

            predicted_proba_vector = self._model.predict_proba(np.array([prediction_feature]))
            predicted_proba = predicted_proba_vector[0]
            for i in range(len(predicted_proba)):
                category = self._le.inverse_transform(np.array([i]))
                outstr += str(category[0]) + ":\t" + format(predicted_proba[i], '.32f') + '\n'
            QMessageBox.about(self, "Prediction", outstr)

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