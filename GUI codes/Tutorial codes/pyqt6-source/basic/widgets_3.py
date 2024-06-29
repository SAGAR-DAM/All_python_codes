import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QCheckBox, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        self.widget = QCheckBox("This is a checkbox")
        self.widget.setCheckState(Qt.CheckState.Checked)

        # For tristate: widget.setCheckState(Qt.PartiallyChecked)
        # Or: widget.setTristate(True)
        self.widget.stateChanged.connect(self.show_state)

        self.setCentralWidget(self.widget)

    def show_state(self, s):
        print(Qt.CheckState(s) == Qt.CheckState.Checked)
        print(s)


def main():
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    app.exec()

if __name__=='__main__':
    main()