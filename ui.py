#!/usr/bin/python

import numpy as np
from PIL import Image
from PyQt4 import QtCore, QtGui

from create_dictionary import KutenDictionary
from classifier import Classifier


###############################
#   QWidget for drawing area  #
###############################


class ScribbleArea(QtGui.QWidget):
    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)

        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 10
        self.myPenColor = QtCore.Qt.black
        self.image = QtGui.QImage()
        self.lastPoint = QtCore.QPoint()

    def openImage(self, fileName):
        loadedImage = QtGui.QImage()
        if not loadedImage.load(fileName):
            return False

        newSize = loadedImage.size().expandedTo(size())
        self.resizeImage(loadedImage, newSize)
        self.image = loadedImage
        self.modified = False
        self.update()
        return True

    def saveImage(self, fileName, fileFormat):
        visibleImage = self.image
        self.resizeImage(visibleImage, size())

        if visibleImage.save(fileName, fileFormat):
            self.modified = False
            return True
        else:
            return False

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.image.fill(QtGui.qRgb(255, 255, 255))
        self.modified = True
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(QtCore.QPoint(0, 0), self.image)

    def resizeEvent(self, event):
        if self.width() > self.image.width() or self.height() > self.image.height():
            newWidth = max(self.width() + 128, self.image.width())
            newHeight = max(self.height() + 128, self.image.height())
            self.resizeImage(self.image, QtCore.QSize(newWidth, newHeight))
            self.update()

        super(ScribbleArea, self).resizeEvent(event)

    def drawLineTo(self, endPoint):
        painter = QtGui.QPainter(self.image)
        painter.setPen(QtGui.QPen(self.myPenColor, self.myPenWidth, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)
        self.modified = True

        rad = self.myPenWidth / 2 + 2
        self.update(QtCore.QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = QtCore.QPoint(endPoint)

    def resizeImage(self, image, newSize):
        if image.size() == newSize:
            return

        newImage = QtGui.QImage(newSize, QtGui.QImage.Format_RGB32)
        newImage.fill(QtGui.qRgb(255, 255, 255))
        painter = QtGui.QPainter(newImage)
        painter.drawImage(QtCore.QPoint(0, 0), image)
        self.image = newImage

    def print_(self):
        printer = QtGui.QPrinter(QtGui.QPrinter.HighResolution)

        printDialog = QtGui.QPrintDialog(printer, self)
        if printDialog.exec_() == QtGui.QDialog.Accepted:
            painter = QtGui.QPainter(printer)
            rect = painter.viewport()
            size = self.image.size()
            size.scale(rect.size(), QtCore.Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.image.rect())
            painter.drawImage(0, 0, self.image)

    def isModified(self):
        return self.modified

    def penColor(self):
        return self.myPenColor

    def penWidth(self):
        return self.myPenWidth




#####################################
#           Main window             #
#####################################


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.saveAsActs = []
        self.scribbleArea = ScribbleArea()
        self.scribbleArea.setFixedSize(500, 500)

        self.createActions()
        self.createMenus()
        
        buttonClassify = QtGui.QPushButton('Classify Character')
        buttonClassify.clicked.connect(self.handleButtonClassify)
        buttonClear = QtGui.QPushButton('Clear Image')
        buttonClear.clicked.connect(self.handleButtonClear)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(buttonClassify)
        hbox.addWidget(buttonClear)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.scribbleArea)
        vbox.addLayout(hbox)

        centralArea = QtGui.QWidget()
        centralArea.setLayout(vbox)

        self.setCentralWidget(centralArea)
        self.setWindowTitle("Japanese Character Classifier")
        self.setWindowIcon(QtGui.QIcon('icon.jpg'))

        self.classify = Classifier()

    def handleButtonClassify(self):
        #self.scribbleArea.image.load('test_images/ka.png')
        self.scribbleArea.update()
        img = self.scribbleArea.image
        img = img.scaled(64, 64, QtCore.Qt.KeepAspectRatio)
        img = img.convertToFormat(QtGui.QImage.Format_RGB888)
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr).reshape(64, 64, 3) / 255.
        arr = arr.dot([0.2, 0.7, 0.1]).reshape(1, 1, 64, 64)
        arr = arr*(-1.) + 1.
        tol = 1.2e-16
        arr[arr<tol] = 0.
        self.classify.classify(arr) # Classify image

    def handleButtonClear(self):
        self.scribbleArea.clearImage()

    def closeEvent(self, event):
        if self.maybeSave():
            event.accept()
        else:
            event.ignore()

    def open(self):
        if self.maybeSave():
            fileName = QtGui.QFileDialog.getOpenFileName(self, "Open File", QtCore.QDir.currentPath())
            if fileName:
                self.scribbleArea.openImage(fileName)

    def save(self):
        action = self.sender()
        fileFormat = action.data()
        self.saveFile(fileFormat)

    def penColor(self):
        newColor = QtGui.QColorDialog.getColor(self.scribbleArea.penColor())
        if newColor.isValid():
            self.scribbleArea.setPenColor(newColor)

    def penWidth(self):
        newWidth, ok = QtGui.QInputDialog.getInteger(self, "Scribble", "Select Rusty's pen width:", self.scribbleArea.penWidth(), 1, 50, 1)
        if ok:
            self.scribbleArea.setPenWidth(newWidth)

    def about(self):
        QtGui.QMessageBox.about(self, "About Scribble",
                "<p>The <b>Japanese Character Classifier</b> demonstrates...</p>")

    def createActions(self):
        self.openAct = QtGui.QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open)

        for format in QtGui.QImageWriter.supportedImageFormats():
            text = "Blablubb!"    
            #text = format.toUpper() + "..."      Okay, wir muessen noch herausfinden, wo hier der Haken liegt!
            action = QtGui.QAction(text, self, triggered=self.save)
            action.setData(format)
            self.saveAsActs.append(action)

        self.printAct = QtGui.QAction("&Print...", self, triggered=self.scribbleArea.print_)
        self.exitAct = QtGui.QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.penColorAct = QtGui.QAction("&Pen Color...", self, triggered=self.penColor)
        self.penWidthAct = QtGui.QAction("Pen &Width...", self, triggered=self.penWidth)
        self.clearScreenAct = QtGui.QAction("&Clear Screen", self, shortcut="Ctrl+L", triggered=self.scribbleArea.clearImage)
        self.aboutAct = QtGui.QAction("&About", self, triggered=self.about)
        self.aboutQtAct = QtGui.QAction("About &Qt", self, triggered=QtGui.qApp.aboutQt)

    def createMenus(self):
        self.saveAsMenu = QtGui.QMenu("&Save As", self)
        for action in self.saveAsActs:
        	self.saveAsMenu.addAction(action)

        fileMenu = QtGui.QMenu("&File", self)
        fileMenu.addAction(self.openAct)
        fileMenu.addMenu(self.saveAsMenu)
        fileMenu.addAction(self.printAct)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAct)

        optionMenu = QtGui.QMenu("&Options", self)
        optionMenu.addAction(self.penColorAct)
        optionMenu.addAction(self.penWidthAct)
        optionMenu.addSeparator()
        optionMenu.addAction(self.clearScreenAct)

        helpMenu = QtGui.QMenu("&Help", self)
        helpMenu.addAction(self.aboutAct)
        helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(fileMenu)
        self.menuBar().addMenu(optionMenu)
        self.menuBar().addMenu(helpMenu)

    def maybeSave(self):
        if self.scribbleArea.isModified():
            ret = QtGui.QMessageBox.warning(self, "Scribble", "The image has been modified.\n Do you want to save your changes?",
                        QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel )
            if ret == QtGui.QMessageBox.Save:
                return self.saveFile('png')
            elif ret == QtGui.QMessageBox.Cancel:
                return False

        return True

    def saveFile(self, fileFormat):
        initialPath = QtCore.QDir.currentPath() + '/untitled.' + fileFormat

        fileName = QtGui.QFileDialog.getSaveFileName(self, "Save As", initialPath, "%s Files (*.%s);;All Files (*)" % (fileFormat.upper(), fileFormat))
        if fileName:
            return self.scribbleArea.saveImage(fileName, fileFormat)

        return False


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



    