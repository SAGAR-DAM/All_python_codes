# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 17:17:59 2023

@author: sagar
"""

import sys

from PyQt6.QtWidgets import (QApplication, 
                             QWidget, 
                             QLineEdit, 
                             QPushButton, 
                             QTextEdit, 
                             QVBoxLayout, 
                             QHBoxLayout,
                             QGridLayout, 
                             QStackedLayout,
                             QMessageBox)

from PyQt6.QtGui import QIcon

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("First app")    # The window name
        self.setWindowIcon(QIcon("D:\\Codes\\GUI codes\\test.png"))   #The app icon
        self.resize(600,600)   # giving the size of the app window (width, height)
        
        layout=QVBoxLayout()
        self.setLayout(layout)
        
        self.inputField=QLineEdit()
        button = QPushButton(text="Press The Button")
        button.clicked.connect(self.sayhellow)
        self.output=QTextEdit()
        
        layout.addWidget(self.inputField)
        layout.addWidget(button)
        layout.addWidget(self.output)
        
    def sayhellow(self):
        inputtext=self.inputField.text()
        self.output.setText(f"Hellow {inputtext} !!!")
        
    #def shutprocess(self):
        #reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        #if reply == QMessageBox.Yes:
            #self.close()
            #print('Window closed')
        #else:
            #pass
        
#app=QApplication([])


def main():
    app=QApplication(sys.argv)
    app.setStyleSheet('''
                      QWidget {font-size: 25px;}
                      QPushButton {font-size: 20px;}
                      
                      
                      ''')

    window=MyApp()
    window.show()

    sys.exit(app.exec())
    
if __name__=='__main__':
    main()