import sys
import PyQt6.QtWidgets.QMainwindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()  # <2>

        self.setWindowTitle("My App")

        self.button = QPushButton("Press Me!",self)  # A pusshbutton with test "Press me" on it
        self.button.resize(100, 50)
        
        '''
        The button.setCheckable is a function to make the button as a checkbox type button or not. 
        
        A. A non-checkbox type button:
        This is something for which the button just have one state. This is a button like a calling bell or keyboard 
        alphabet button (not the caps lock). The button is only on at the pressing time. That is the "True" state of
        the button.  As soon as the button is released, the button is off or in "False" state. for this type button
        the clicked sends the checked message which is always false. So when we make the button.setCheckable(False),
        i.e. the button as a non checkbox type button, and press it, we get the false as the retur. By default the
        button is always in False mode.
        
        
        B. A checkbox type button: 
        This s something like a switchboard switch or a checkbox button or a better examplr is the caps lock on a
        keyboard. The buttons have two states. On (or True) and off (or False). The Button will flip the state as 
        soon as it is pressed. Also it will stay in the state and not flip automatically until it is pressed again.
        The caps-lock has to be pressed to turned on and then we can type anything it will be in caps mode. After 
        that to get back the previous mode, we need to press the caps-lock again to turn of the mdoe. The two states
        of this buttons are following:
        
            1. True (or when the button is on): This state is like a checkbox after clicking the tick sign.
            or like a normal switchboard switch after it is on. The button will keep the same state till further 
            pressing... Like the fan will be on until the switch is pressed again to off (here false) state. Or the 
            checkbox keeps the tick mark until we uncheck it again by pressing the box.
            
            2. False (or when the button is off): This is the reverse of the previous state. And this state also keeps
            the same state until the button is further pressed. Like if the checkbox is deselected, then we have to 
            again selsect it to keep it active. It will not get activated unless we press it.
        '''
        self.button.setCheckable(True)   # will make the button checkable and with initial state as "True"
        
        
        '''
        The "True" inside setChekable is nothing with the state of the checkable button.
        It only makes the button from a normal to checkable one. After that the default state of the button is TRUE
        
        The toggle is a function to reverse the current state of the button.So if it's on, it will turn it off
        and if it's off, then will tuen on.
        
        we can use the toggle anytime to change the button state without pressing it.
        '''
        #button.toggle()  #will reverse the state of the checkable button
        
        
        
        
        
        '''These are the functions to connect the button to an event. The functions has multiple output.
           one of them is the checked state. if the button is not checkable then the default o/p is False.
           For a checkable button it will give the state and reverse the current state as the button was clicked.
        '''
        self.button.clicked.connect(self.the_button_was_clicked)
        self.button.clicked.connect(self.the_button_was_toggled)
        
        #button.toggle()
        
        # Set the central widget of the Window.
        #self.setCentralWidget(self.button)
        
        
        self.buttoncount=0  # A variable to count the number of presses on the button
    def the_button_was_clicked(self, checked):
        print("Clicked!")
        self.button.setText(f"{checked} \ncount: {1+self.buttoncount}")   # Will show the text on the button as the state of the button, will change in each press 
        self.buttoncount+=1
    
    def the_button_was_toggled(self, checked):
        print("Checked?", bool(checked))
        if(bool(checked)==True):
            print("The QPushbutton is in true state\n")
            self.button.setStyleSheet("QPushButton {background-color:green}")
        else:
            print("The QPusbutton is in false state\n")
            self.button.setStyleSheet("QPushButton {background-color:red}")
def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
    
#sys.modules[__name__].__dict__.clear()