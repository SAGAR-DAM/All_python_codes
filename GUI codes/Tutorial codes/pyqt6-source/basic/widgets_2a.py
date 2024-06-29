import sys

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        self.widget = QLabel("Hello")
        self.widget.setPixmap(QPixmap("D:\\wallpapers\\hdfc bank.jpg"))
        #self.widget.setText("HDFC BANK")
        
        self.setCentralWidget(self.widget)



def main():
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    app.exec()

if __name__=='__main__':
    main()