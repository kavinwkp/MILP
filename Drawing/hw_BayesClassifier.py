import sys
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox, QFont
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter, \
    QComboBox, QLabel, QSpinBox, QFileDialog, QApplication
from hw_PaintBoard import PaintBoard
from NaiveBayesScratch import NaiveBayesScratch
from PIL import Image, ImageQt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PyQt5.QtCore import Qt, QRect
# import cv2


class MainWidget(QWidget):
    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.__result = -1
        self.__xtest = []
        self.__ytest = []
        self.__initData()
        self.__initView()

    def __initData(self):
        self.__paintBoard = PaintBoard(self)
        mint = tf.keras.datasets.mnist
        (X, y), (X_test, y_test) = mint.load_data()
        self.model = NaiveBayesScratch()
        X = np.array(X).reshape(60000, 784)
        self.__xtest = np.array(X_test).reshape(10000, 784)
        self.__ytest = y_test
        self.model.fit(X, y)

    def __initView(self):
        self.setFixedSize(600, 400)
        self.setWindowTitle('Application')

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.addWidget(self.__paintBoard)

        sub_layout = QVBoxLayout()
        sub_layout.setContentsMargins(10, 10, 10, 10)
        sub_layout.setSpacing(30)

        self.__btn_Clear = QPushButton('clear')
        self.__btn_Clear.setParent(self)
        self.__btn_Clear.clicked.connect(self.__paintBoard.clear)
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_Predict = QPushButton('predict')
        self.__btn_Predict.setParent(self)
        self.__btn_Predict.clicked.connect(self.predict)
        sub_layout.addWidget(self.__btn_Predict)


        self.__btn_Quit = QPushButton('quit')
        self.__btn_Quit.setParent(self)
        self.__btn_Quit.clicked.connect(self.quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__lb_Result_Tip = QLabel()
        font = QFont()
        font.setPointSize(24)
        self.__lb_Result_Tip.setFont(font)

        self.__lb_Result_Tip.setText('result')
        self.__lb_Result_Tip.setParent(self)
        sub_layout.addWidget(self.__lb_Result_Tip)

        self.__lb_Result = QLabel()
        font = QFont()
        font.setPointSize(30)
        self.__lb_Result.setFont(font)
        self.__lb_Result.setParent(self)
        self.__lb_Result.setAlignment(Qt.AlignHCenter)
        sub_layout.addWidget(self.__lb_Result)

        main_layout.addLayout(sub_layout)

    def quit(self):
        self.close()

    def predict(self):
        image = self.__paintBoard.getImage()
        pil_img = ImageQt.fromqimage(image)
        pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)
        # pil_img.save('./images/test66.png')
        # pil_img.show()
        # img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_BGR2GRAY)
        # img = cv2.medianBlur(img, 3)
        # cv2.imshow("img", img)

        img_array = np.array(pil_img.convert('L')).reshape(784)
        # print(img_array.shape)      # (784,)

        # display image
        # print(img_array)
        # plt.imshow(img_array.reshape(28, 28), cmap="binary")
        # plt.imshow(pil_img, cmap="binary")
        # plt.show()
        # fig = plt.figure(figsize=(6, 6))
        # fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        # # 绘制数字：每张图像8*8像素点
        # for i in range(64):
        #     ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        #     ax.imshow(self.xtest[i].reshape(28, 28), cmap=plt.cm.binary, interpolation='nearest')
        #     # 用目标值标记图像
        #     ax.text(0, 7, str(self.ytest[i]))
        # plt.show()

        self.__result = self.model.predict(img_array)
        print("result: %d" % self.__result)
        self.__lb_Result.setText("%d" % self.__result)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./images/draw.png'))
    m = MainWidget()
    m.show()
    sys.exit(app.exec_())
