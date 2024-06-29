#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:12:50 2023

@author: sagar
"""

''' This is a basic calculator program with PyQt6'''


import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLineEdit, QPushButton

from PyQt6.QtGui import QIcon

class Calculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Sagar's Calculator")
        self.setWindowIcon(QIcon("D:\\Codes\\GUI codes\\calculator.png"))   #The app icon
        self.setGeometry(100, 100, 300, 250)
        

        # Create input field
        self.input = QLineEdit(self)
        self.input.move(10, 10)
        self.input.resize(280, 40)

        # Create buttons
        self.button1 = QPushButton("1", self)
        self.button1.move(10, 60)
        self.button1.resize(30, 30)
        self.button1.clicked.connect(lambda: self.numberPressed("1"))
        self.button1.setStyleSheet("QPushButton {background-color:magenta}")

        self.button2 = QPushButton("2", self)
        self.button2.move(60, 60)
        self.button2.resize(30, 30)
        self.button2.clicked.connect(lambda: self.numberPressed("2"))
        self.button2.setStyleSheet("QPushButton {background-color:yellow}")

        self.button3 = QPushButton("3", self)
        self.button3.move(110, 60)
        self.button3.resize(30, 30)
        self.button3.clicked.connect(lambda: self.numberPressed("3"))
        self.button3.setStyleSheet("QPushButton {background-color:grey}")

        self.button4 = QPushButton("4", self)
        self.button4.move(10, 110)
        self.button4.resize(30, 30)
        self.button4.clicked.connect(lambda: self.numberPressed("4"))
        self.button4.setStyleSheet("QPushButton {background-color:cyan}")

        self.button5 = QPushButton("5", self)
        self.button5.move(60, 110)
        self.button5.resize(30, 30)
        self.button5.clicked.connect(lambda: self.numberPressed("5"))
        self.button5.setStyleSheet("QPushButton {background-color:brown}")

        self.button6 = QPushButton("6", self)
        self.button6.move(110, 110)
        self.button6.resize(30, 30)
        self.button6.clicked.connect(lambda: self.numberPressed("6"))
        self.button6.setStyleSheet("QPushButton {background-color:pink}")

        self.button7 = QPushButton("7", self)
        self.button7.move(10, 160)
        self.button7.resize(30, 30)
        self.button7.clicked.connect(lambda: self.numberPressed("7"))
        self.button7.setStyleSheet("QPushButton {background-color:violet}")

        self.button8 = QPushButton("8", self)
        self.button8.move(60, 160)
        self.button8.resize(30, 30)
        self.button8.clicked.connect(lambda: self.numberPressed("8"))
        self.button8.setStyleSheet("QPushButton {background-color:grey}")

        self.button9 = QPushButton("9", self)
        self.button9.move(110, 160)
        self.button9.resize(30, 30)
        self.button9.clicked.connect(lambda: self.numberPressed("9"))
        self.button9.setStyleSheet("QPushButton {background-color:yellow}")

        self.button0 = QPushButton("0", self)
        self.button0.move(60, 210)
        self.button0.resize(30, 30)
        self.button0.clicked.connect(lambda: self.numberPressed("0"))
        self.button0.setStyleSheet("QPushButton {background-color:cyan}")

        self.buttonPlus = QPushButton("+", self)
        self.buttonPlus.move(160, 60)
        self.buttonPlus.resize(30, 30)
        self.buttonPlus.clicked.connect(lambda: self.operationPressed("+"))
        self.buttonPlus.setStyleSheet("QPushButton {background-color:orange}")

        self.buttonMinus = QPushButton("-", self)
        self.buttonMinus.move(160, 110)
        self.buttonMinus.resize(30, 30)
        self.buttonMinus.clicked.connect(lambda: self.operationPressed("-"))
        self.buttonMinus.setStyleSheet("QPushButton {background-color:orange}")

        self.buttonMultiply = QPushButton("*", self)
        self.buttonMultiply.move(160, 160)
        self.buttonMultiply.resize(30, 30)
        self.buttonMultiply.clicked.connect(lambda: self.operationPressed("*"))
        self.buttonMultiply.setStyleSheet("QPushButton {background-color:orange}")

        self.buttonDivide = QPushButton("/", self)
        self.buttonDivide.move(160, 210)
        self.buttonDivide.resize(30, 30)
        self.buttonDivide.clicked.connect(lambda: self.operationPressed("/"))
        self.buttonDivide.setStyleSheet("QPushButton {background-color:orange}")

        self.buttonEquals = QPushButton("=", self)
        self.buttonEquals.move(210, 160)
        self.buttonEquals.resize(30, 30)
        self.buttonEquals.clicked.connect(self.equalsPressed)
        self.buttonEquals.setStyleSheet("QPushButton {background-color:green}")
        #self.buttonEquals.clicked.connect(lambda: self.operationPressed("Enter"))

        self.buttonClear = QPushButton("C", self)
        self.buttonClear.move(210, 110)
        self.buttonClear.resize(30, 30)
        self.buttonClear.clicked.connect(self.clearPressed)
        self.buttonClear.setStyleSheet("QPushButton {background-color:green}")
        
        self.buttonDecimal = QPushButton(".", self)
        self.buttonDecimal.move(110, 210)
        self.buttonDecimal.resize(30, 30)
        self.buttonDecimal.clicked.connect(lambda: self.numberPressed("."))
        self.buttonDecimal.setStyleSheet("QPushButton {background-color:yellow}")

        self.show()

    def numberPressed(self, number):
        currentText = self.input.text()
        newText = currentText + number
        self.input.setText(newText)

    def operationPressed(self, operation):
        currentText = self.input.text()
        newText = currentText + " " + operation + " "
        self.input.setText(newText)

    def equalsPressed(self):
        currentText = self.input.text()
        try:
            result = eval(currentText)
            self.input.setText(str(result))
        except:
            self.input.setText("Error")

    def clearPressed(self):
        self.input.setText("")



def main():
    app = QApplication(sys.argv)

    window = Calculator()
    window.show()

    sys.exit(app.exec())

if __name__=='__main__':
    main()    