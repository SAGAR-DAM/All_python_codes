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
        time.sleep(0.01)
        # Update progress bar and set postfix to show additional info
        pbar.set_postfix({"Loss": f"{(100 - i) / 100:.4f}"})  # Example for loss
        pbar.update(1)
        
        
        
# Foreground (text) colors
BLACK = "\033[30m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"

# Background colors
BG_BLACK = "\033[40m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"

# Reset color
RESET = "\033[0m"

# Printing text in different colors
print(f"{BLACK}This is black text{RESET}")
print(f"{RED}This is red text{RESET}")
print(f"{GREEN}This is green text{RESET}")
print(f"{YELLOW}This is yellow text{RESET}")
print(f"{BLUE}This is blue text{RESET}")
print(f"{MAGENTA}This is magenta text{RESET}")
print(f"{CYAN}This is cyan text{RESET}")
print(f"{WHITE}This is white text{RESET}")

# Printing text with different background colors
print(f"{BG_BLACK}{WHITE}This is black background{RESET}")
print(f"{BG_RED}{WHITE}This is red background{RESET}")
print(f"{BG_GREEN}{BLACK}This is green background{RESET}")
print(f"{BG_YELLOW}{BLACK}This is yellow background{RESET}")
print(f"{BG_BLUE}{WHITE}This is blue background{RESET}")
print(f"{BG_MAGENTA}{WHITE}This is magenta background{RESET}")
print(f"{BG_CYAN}{BLACK}This is cyan background{RESET}")
print(f"{BG_WHITE}{BLACK}This is white background{RESET}")
