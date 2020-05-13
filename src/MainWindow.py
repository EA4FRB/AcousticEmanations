# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ui\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(906, 963)
        MainWindow.setAccessibleName("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBoxTrain = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxTrain.setGeometry(QtCore.QRect(10, 0, 221, 221))
        self.groupBoxTrain.setObjectName("groupBoxTrain")
        self.lineEditKeyVal = QtWidgets.QLineEdit(self.groupBoxTrain)
        self.lineEditKeyVal.setGeometry(QtCore.QRect(10, 40, 113, 22))
        self.lineEditKeyVal.setObjectName("lineEditKeyVal")
        self.spinBoxRecNb = QtWidgets.QSpinBox(self.groupBoxTrain)
        self.spinBoxRecNb.setGeometry(QtCore.QRect(10, 80, 42, 22))
        self.spinBoxRecNb.setObjectName("spinBoxRecNb")
        self.labelFileName = QtWidgets.QLabel(self.groupBoxTrain)
        self.labelFileName.setGeometry(QtCore.QRect(10, 110, 121, 16))
        self.labelFileName.setObjectName("labelFileName")
        self.pushButtonRec = QtWidgets.QPushButton(self.groupBoxTrain)
        self.pushButtonRec.setGeometry(QtCore.QRect(10, 160, 131, 28))
        self.pushButtonRec.setObjectName("pushButtonRec")
        self.progressBarRecord = QtWidgets.QProgressBar(self.groupBoxTrain)
        self.progressBarRecord.setGeometry(QtCore.QRect(10, 190, 131, 23))
        self.progressBarRecord.setProperty("value", 24)
        self.progressBarRecord.setObjectName("progressBarRecord")
        self.labelKeyName = QtWidgets.QLabel(self.groupBoxTrain)
        self.labelKeyName.setGeometry(QtCore.QRect(10, 20, 111, 16))
        self.labelKeyName.setObjectName("labelKeyName")
        self.labelIndex = QtWidgets.QLabel(self.groupBoxTrain)
        self.labelIndex.setGeometry(QtCore.QRect(10, 60, 111, 16))
        self.labelIndex.setObjectName("labelIndex")
        self.groupBoxModel = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxModel.setGeometry(QtCore.QRect(10, 220, 881, 661))
        self.groupBoxModel.setObjectName("groupBoxModel")
        self.MplWidgetMatrix = MplWidget(self.groupBoxModel)
        self.MplWidgetMatrix.setGeometry(QtCore.QRect(0, 20, 871, 541))
        self.MplWidgetMatrix.setObjectName("MplWidgetMatrix")
        self.pushButtonModel = QtWidgets.QPushButton(self.groupBoxModel)
        self.pushButtonModel.setGeometry(QtCore.QRect(730, 570, 131, 28))
        self.pushButtonModel.setObjectName("pushButtonModel")
        self.labelTrainAccuracy = QtWidgets.QLabel(self.groupBoxModel)
        self.labelTrainAccuracy.setGeometry(QtCore.QRect(480, 580, 241, 16))
        self.labelTrainAccuracy.setObjectName("labelTrainAccuracy")
        self.labelTestAccuracy = QtWidgets.QLabel(self.groupBoxModel)
        self.labelTestAccuracy.setGeometry(QtCore.QRect(480, 600, 241, 16))
        self.labelTestAccuracy.setObjectName("labelTestAccuracy")
        self.labelPreTrainAccuracy = QtWidgets.QLabel(self.groupBoxModel)
        self.labelPreTrainAccuracy.setGeometry(QtCore.QRect(480, 560, 241, 16))
        self.labelPreTrainAccuracy.setObjectName("labelPreTrainAccuracy")
        self.labelFeatures = QtWidgets.QLabel(self.groupBoxModel)
        self.labelFeatures.setGeometry(QtCore.QRect(10, 570, 451, 41))
        self.labelFeatures.setTextFormat(QtCore.Qt.RichText)
        self.labelFeatures.setWordWrap(True)
        self.labelFeatures.setObjectName("labelFeatures")
        self.progressBarCompute = QtWidgets.QProgressBar(self.groupBoxModel)
        self.progressBarCompute.setGeometry(QtCore.QRect(730, 600, 131, 23))
        self.progressBarCompute.setProperty("value", 24)
        self.progressBarCompute.setObjectName("progressBarCompute")
        self.pushButtonTest = QtWidgets.QPushButton(self.groupBoxModel)
        self.pushButtonTest.setGeometry(QtCore.QRect(10, 620, 93, 28))
        self.pushButtonTest.setObjectName("pushButtonTest")
        self.MplWidget = MplWidget(self.centralwidget)
        self.MplWidget.setGeometry(QtCore.QRect(230, 0, 311, 231))
        self.MplWidget.setObjectName("MplWidget")
        self.MplWidgetSpectrum = MplWidget(self.centralwidget)
        self.MplWidgetSpectrum.setGeometry(QtCore.QRect(540, 0, 351, 231))
        self.MplWidgetSpectrum.setObjectName("MplWidgetSpectrum")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 906, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionFolderSelect = QtWidgets.QAction(MainWindow)
        self.actionFolderSelect.setObjectName("actionFolderSelect")
        self.actionLoadWaveFile = QtWidgets.QAction(MainWindow)
        self.actionLoadWaveFile.setObjectName("actionLoadWaveFile")
        self.menuFile.addAction(self.actionLoadWaveFile)
        self.menuFile.addAction(self.actionFolderSelect)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Acoustic Emanations Tool"))
        self.groupBoxTrain.setTitle(_translate("MainWindow", "Record"))
        self.labelFileName.setText(_translate("MainWindow", "TextLabel"))
        self.pushButtonRec.setText(_translate("MainWindow", "Record"))
        self.labelKeyName.setText(_translate("MainWindow", "Key name:"))
        self.labelIndex.setText(_translate("MainWindow", "Index:"))
        self.groupBoxModel.setTitle(_translate("MainWindow", "Model"))
        self.pushButtonModel.setText(_translate("MainWindow", "Run Model"))
        self.labelTrainAccuracy.setText(_translate("MainWindow", "Train accuracy:"))
        self.labelTestAccuracy.setText(_translate("MainWindow", "Test accuracy:"))
        self.labelPreTrainAccuracy.setText(_translate("MainWindow", "Pre-train accuracy:"))
        self.labelFeatures.setText(_translate("MainWindow", "Features:"))
        self.pushButtonTest.setText(_translate("MainWindow", "Test"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionFolderSelect.setText(_translate("MainWindow", "Folder Select"))
        self.actionLoadWaveFile.setText(_translate("MainWindow", "Load wave file"))
from mplwidget import MplWidget
