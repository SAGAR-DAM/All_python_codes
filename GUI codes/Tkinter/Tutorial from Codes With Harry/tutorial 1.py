# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 13:25:33 2023

@author: sagar
"""

import tkinter as tk
#from skimage.transform import resize
#from skimage import io
#from PIL import Image

root=tk.Tk()

root.geometry("600x400")
root.title("Sagar GUI")

myimage=tk.PhotoImage(file="D:\\Codes\\GUI codes\\test.png")


image=tk.Label(image=myimage)
image.pack()

# Important Label Options
# text - adds the text
# bd - background
# fg - foreground
# font - sets the font
# 1. font=("comicsansms", 19, "bold")
# 2. font="comicsansms 19 bold"

# padx - x padding
# pady - y padding
# relief - border styling - SUNKEN, RAISED, GROOVE, RIDGE

title_label = tk.Label(text ='''Abdul Rashid Salim Salman Khan is an Indian film actor, producer, 
                       occasional playback singer and television personality. In a film career spanning almost thirty years, 
                       Khan has received numerous awards, including two National Film Awards as a film producer, and two 
                       Filmfare Awards for acting. He has a significant following in Asia and the Indian diaspora worldwide, 
                       and is cited in the media as one of the most commercially successful actors of Indian cinema. According 
                       to the Forbes 2018 list of Top-Paid 100 Celebrity Entertainers in world, Khan was the highest ranked Indian 
                       with 82nd rank with earnings of $37.7 million.'''
                       , bg ="red", fg="white", padx=13, pady=94, font=("Times New Roman", 9, "bold"), borderwidth=3, relief=tk.SUNKEN)

# Important Pack options
# anchor = nw
# side = top, bottom, left, right
# fill
# padx
# pady

title_label.pack(side=tk.TOP, anchor ="nw")
#title_label.pack()

root.mainloop()
