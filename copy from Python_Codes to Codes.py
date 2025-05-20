# -*- coding: utf-8 -*-
"""
Created on Sun May  4 09:59:09 2025

@author: mrsag
"""

import os
import shutil

def copy_selected_files(src, dst, extensions=(".py", ".ipynb", ".cpp", ".c", "f77", ".jl", ".java")):
    for root, _, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        dest_dir = os.path.join(dst, rel_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        for file in files:
            if file.endswith(extensions):
                shutil.copy2(os.path.join(root, file), os.path.join(dest_dir, file))

# Example usage:
source = r"D:\data Lab"
destination = r"D:\All lab codes"

copy_selected_files(source, destination)
