import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen, \
    QColor, QSize
from PyQt5.QtCore import Qt

class PaintBoard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__InitData()
        self.__InitView()

    def __InitData(self):
        self.__size = QSize(280, 280)
        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.black)
        self.__isEmpty = True
        self.__lastPose = QPoint(0, 0)
        self.__currentPose = QPoint(0, 0)
        self.__painter = QPainter()
        self.__thickness = 24
        self.__penColor = QColor('white')

    def __InitView(self):
        self.setFixedSize(self.__size)

    def paintEvent(self, paintEvent):
        self.__painter.begin(self)
        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouseEvent):
        self.__currentPose = mouseEvent.pos()
        self.__lastPose = self.__currentPose

    def mouseMoveEvent(self, mouseEvent):
        self.__currentPose = mouseEvent.pos()
        self.__painter.begin(self.__board)
        self.__painter.setPen(QPen(self.__penColor, self.__thickness))
        self.__painter.drawLine(self.__lastPose, self.__currentPose)
        self.__painter.end()
        self.__lastPose = self.__currentPose
        self.update()

    def mouseReleaseEvent(self, mouseEvent):
        self.__isEmpty = False

    def clear(self):
        self.__board.fill(Qt.black)
        self.update()
        self.__isEmpty = True

    def isEmpty(self):
        return self.__isEmpty

    def getImage(self):
        image = self.__board.toImage()
        return image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    m = PaintBoard()
    m.show()
    sys.exit(app.exec_())


