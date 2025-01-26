#!/usr/bin/python3
import random
import time
from tkinter import *

# Set a global time delay for animation in seconds (can be adjusted)
time_of_sleep = 0.1

class VisualTower:
    def __init__(self, n, p0, p1, p2, p3):
        self.myGui = Tk()
        self.myGui.title("Tower Of Hanoi")
        
        # Initialize pegs and frames for each tower
        self.p = [p0, p1, p2, p3]
        self.Frames = []
        self._n = n
        for i in range(4):
            frame = Frame(self.myGui, height=self._n * 100, width=(self._n + 1) * 25)
            frame.grid(row=0, column=i)
            self.Frames.append(frame)
        self.showit()
        self.myGui.update()
        self.myGui.quit()
    
    def showit(self):
        # Draw each peg and disks on the frame
        w = [None] * self._n  # Create list of canvases for each disk
        for f in range(4):
            base = Canvas(self.Frames[f], width=self._n * 25, height=26)
            base.grid(row=self._n, column=0, columnspan=(self._n + 1))
            base.create_rectangle(0, 0, self._n * 25, 10, fill="blue")
            
            for i in range(0, self.p[f]._size):
                w[i] = Canvas(self.Frames[f], width=self.p[f].stk[i] * 25, height=26)
                w[i].grid(row=self._n - i - 1, column=0, columnspan=(self._n - self.p[f].stk[i] + 1))
                w[i].create_rectangle(0, 0, self.p[f].stk[i] * 25, 25, fill="black")
            
            for i in range(self.p[f]._size, self._n):
                w[i] = Canvas(self.Frames[f], width=self._n * 25, height=26)
                w[i].grid(row=self._n - i - 1, column=0, columnspan=(self._n + 1))
                w[i].create_rectangle(0, 0, self._n * 25, 25, fill="white")
        
        self.myGui.update()
        if self.p[3]._size == self._n and self.p[0]._size == 0:
            self.myGui.mainloop()

class Stack:
    def __init__(self, size, name):
        self.name = name
        self._size = size
        self.stk = list(range(self._size, 0, -1))
    
    def pop(self):
        if self._size > 0:
            _tmp = self.stk[self._size - 1]  # Element to be popped
            self.stk = self.stk[:self._size - 1]
            self._size -= 1
            return _tmp
        else:
            return -1
    
    def push(self, elem):
        if self._size == 0 or elem < self.stk[self._size - 1]:
            self.stk.append(elem)
            self._size += 1
        else:
            print("This Operation is Invalid in Tower's of Hanoi ;)")

class Hanoi:
    def __init__(self, n):
        self.n = n
        self.p1 = Stack(self.n, "peg1")
        self.p2 = Stack(self.n, "peg2")
        self.p3 = Stack(self.n, "peg3")
        self.p4 = Stack(self.n, "peg4")
        self.vt = VisualTower(self.n, self.p1, self.p2, self.p3, self.p4)
        
        # Clear peg 2, 3, and 4 initially
        while self.p2.pop() != -1:
            pass
        while self.p3.pop() != -1:
            pass
        while self.p4.pop() != -1:
            pass
        self.displayit()
    
    def displayit(self):
        self.vt.showit()
        time.sleep(time_of_sleep)

def tower_of_hanoi(num_of_Disks, src, inter1, inter2, dest):
    global H, time_of_sleep
    if num_of_Disks == 1:
        print(f"\nTransfer Disc from {src.name} >> TO >> {dest.name}")
        dest.push(src.pop())
        H.displayit()
    elif num_of_Disks == 2:
        print(f"\nTransfer Disc from {src.name} >> TO >> {inter1.name}")
        inter1.push(src.pop())
        H.displayit()
        print(f"\nTransfer Disc from {src.name} >> TO >> {dest.name}")
        dest.push(src.pop())
        H.displayit()
        print(f"\nTransfer Disc from {inter1.name} >> TO >> {dest.name}")
        dest.push(inter1.pop())
        H.displayit()
    else:
        tower_of_hanoi(num_of_Disks - 2, src, inter2, dest, inter1)
        print(f"\nTransfer Disc from {src.name} >> TO >> {inter2.name}")
        inter2.push(src.pop())
        H.displayit()
        print(f"\nTransfer Disc from {src.name} >> TO >> {dest.name}")
        dest.push(src.pop())
        H.displayit()
        print(f"\nTransfer Disc from {inter2.name} >> TO >> {dest.name}")
        dest.push(inter2.pop())
        H.displayit()
        tower_of_hanoi(num_of_Disks - 2, inter1, src, inter2, dest)

def main():
    n = int(input("No. Of Disks: "))
    global H
    H = Hanoi(n)
    tower_of_hanoi(n, H.p1, H.p2, H.p3, H.p4)

if __name__ == "__main__":
    main()
