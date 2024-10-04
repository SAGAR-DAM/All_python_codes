# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 00:07:00 2024

@author: mrsag
"""

from tqdm import tqdm
import time

# ANSI color codes
GREEN = "\033[92m"
RESET = "\033[0m"

# Custom progress bar with color and style
total_iterations = 100
with tqdm(total=total_iterations, 
          desc="Processing",  # Normal description
          ncols=100, 
          bar_format=f"{{l_bar}}{GREEN}{{bar}}{RESET}|{{n_fmt}}/{{total_fmt}} [{GREEN}{{percentage:.0f}}%{RESET}]{{postfix}}") as pbar:  # Include postfix
    for i in range(total_iterations):
        time.sleep(0.05)
        # Update progress bar and set postfix to show additional info
        pbar.set_postfix({"Loss": f"{(100 - i) / 100:.4f}"})  # Example for loss
        pbar.update(1)