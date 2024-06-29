# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:21:58 2023

@author: sagar
"""

"""
if we have a (10,10,10) matrix and say we need the position of the number closest 
to the number 5.1 which is positionwise sitting closest to the index [2,3,5].

say the closest number of 5.1 in the array is 5.15
But there might be the scene that 5.15 is present in multiple positions in the array.

among all the 5.15s present say location [3,4,6] is closest to the given position 
i.e. [2,3,5].

the function will return the values:

closest_val: 5.15
mindist_index_of_closest_val_in_arr: [3,4,6]
"""

import numpy as np
x=np.random.uniform(low=1,high=10, size=(8,8))
x=np.round(x)


x[4,3]=6.2
'''
x[4,4]=5
x[3,4]=5
x[3,3]=5
x[4,2]=5
x[5,2]=5
x[7,1]=5
x[7,3]=5
'''
print(x)


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (abs(array - value)).argmin()
    val=array.flat[idx]
    index=np.where(array==val)
    return val,index

def dist_in_array(index1,index2):
    dist=0
    for i in range(len(index2)):
        dist+=(index2[i]-index1[i])**2
    return(dist)
    
def closest_index(array,value,old_index):
    closest_val,closest_val_indices=find_nearest_value(array,value)
    #print(f"val: {val}")
    #print(f"index: {index}")
    dist=[]
    for i in range(len(closest_val_indices[0])):
        index2=[]
        for j in range(len(old_index)):
            index2.append(closest_val_indices[j][i])
        #print(f"index2: {index2}")
        dist.append(dist_in_array(old_index,index2))
    
    mindist,mindist_index=find_nearest_value(dist,np.min(dist))
    #print(f"mindist: {mindist}")
    #print(f"mindist_index: {mindist_index}")
    mindist_index_of_closest_val_in_arr=[]
    
    for i in range(len(old_index)):
        #print(mindist_index[0][0])
        mindist_index_of_closest_val_in_arr.append(closest_val_indices[i][mindist_index[0][0]])
    return(closest_val,mindist_index_of_closest_val_in_arr)
    
closest_val,mindist_index_of_closest_val_in_arr=closest_index(array=x, value=5.1, old_index=[4,3])
print(closest_val)
print(mindist_index_of_closest_val_in_arr)
#print(dist_in_array([0,0],[5,9]))