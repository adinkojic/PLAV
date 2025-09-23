"""ion remember"""

from PyQt6 import QtWidgets, QtCore, QtGui

app = QtWidgets.QApplication([])
avonics_window = QtWidgets.QMainWindow()

roll = 5

avonics_window.setWindowTitle('Instruments')
avonics_window.setFixedSize(1000, 1000)

pixmap = QtGui.QPixmap("assets/attitudebg.png")
instrument_widget = QtWidgets.QWidget()
controls_layout = QtWidgets.QVBoxLayout()

label = QtWidgets.QLabel()
label.setPixmap(pixmap)
avonics_window.setCentralWidget(label)

painter = QtGui.QPainter()
painter.drawPixmap(0, 0, pixmap)
painter.save()
cx, cy = avonics_window.width()/2, avonics_window.height()/2
painter.translate(cx, cy)
painter.rotate(-roll)

painter.drawPixmap(QtCore.QPointF(), pixmap)
painter.restore()

avonics_window.show()


app.exec()
