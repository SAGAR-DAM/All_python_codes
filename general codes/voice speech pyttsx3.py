# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:00:58 2024

@author: mrsag
"""

import pyttsx3

engine = pyttsx3.init()
engine.say("hey your program has been completed !!")
engine.runAndWait()

del(engine)