#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# Step 1: Create a 3D data cube (Location x Time x Item Types)

data_cube = np.array([
    # Chennai, Kolkata, Mumbai, Delhi
    [[340, 435], [360, 460], [20, 20], [10, 15]],  # Q1
    [[390, 385], [20, 39], [50, 35], [260, 508]],  # Q2
    [[15, 60], [48, 48], [38, 35], [390, 256]],    # Q3
    [[20, 90], [39, 80], [436, 396], [50, 40]]     # Q4
])


locations = ["Chennai", "Kolkata", "Mumbai", "Delhi"]
quarters = ["Q1", "Q2", "Q3", "Q4"]
item_types = ["Type A", "Type B"]

# Step 2: Implement operations on the data cube

# Roll-up: Aggregate data across quarters (sum of quarters for each location and item type)
def roll_up(data):
    return np.sum(data, axis=0)  # Aggregate over time

rollup_result = roll_up(data_cube)

# Drill-down: Go deeper into specific quarters
def drill_down(data, quarter_index):
    return data[quarter_index, :, :]  # Specific quarter details

drilldown_result = drill_down(data_cube, 0)  

# Slicing: Filter data for a specific location
def slice_data(data, location_index):
    return data[:, location_index, :]  
slicing_result = slice_data(data_cube, 2)  

# Dicing: Filter data for a subcube (specific quarters and locations)
def dice_data(data, quarter_indices, location_indices):
    return data[np.ix_(quarter_indices, location_indices, range(data.shape[2]))]

dicing_result = dice_data(data_cube, [1, 2], [0, 3])  
# Pivot: Change the view of the data cube
def pivot_data(data):
    reshaped = data.transpose(1, 0, 2)  # Swap axes (e.g., Location x Quarter x Item Types)
    return reshaped

pivot_result = pivot_data(data_cube)

# Step 3: Display results
print("Original Data Cube (Shape):", data_cube.shape)
print("\nRoll-up Result (Aggregated over Quarters):\n", rollup_result)
print("\nDrill-down Result (Details for Q1):\n", drilldown_result)
print("\nSlicing Result (All Quarters for Mumbai):\n", slicing_result)
print("\nDicing Result (Q2 and Q3 for Chennai and Delhi):\n", dicing_result)
print("\nPivot Result (Swapped Dimensions):\n", pivot_result)


# In[ ]:




