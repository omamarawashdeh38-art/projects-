#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# 3D points representing the human pose
points = np.array([
    [0, 0, 0],     # P1
    [6.5, 2.5, -2.5],  # P2
    [6.5, 2.5, -7.5],  # P3
    [6.5, -2.5, -2.5],  # P4
    [6.5, -2.5, -7.5],  # P5
    [0, 0, 10],     # P6
    [0, 0, 12.5],   # P7
    [0, 2.5, 5],    # P8
    [5, 2.5, 5],    # P9
    [0, -2.5, 5],   # P10
    [5, -2.5, 5]    # P11
])

# Indices representing body parts
larm_idx = [0, 5, 7, 8]
rarm_idx = [0, 5, 9, 10]
lleg_idx = [0, 1, 2]
rleg_idx = [0, 3, 4]
torso_idx = [0, 5]
head_idx = [5, 6]

body_parts = [larm_idx, rarm_idx, lleg_idx, rleg_idx, torso_idx, head_idx]

# Camera parameters
camera_distance = 25
camera_positions = [camera_distance * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0])
                    for angle in range(0, 360, 45)]
image_plane_z = 5  # Projection plane

# Perspective projection function
def perspective_projection(points, camera_position):
    """
    Projects 3D points onto a 2D plane using perspective projection.
    """
    # Translate points relative to the camera
    relative_points = points - camera_position
    
    # Project onto 2D plane
    projected_points = relative_points[:, :2] / (relative_points[:, 2:3] + image_plane_z)
    return projected_points

# Visualization of projected images
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
axs = axs.ravel()

for i, camera_pos in enumerate(camera_positions):
    # Perform projection
    projected_points = perspective_projection(points, camera_pos)
    
    # Plot projected pose
    ax = axs[i]
    ax.set_title(f"Camera @ {45 * i}°")
    for part in body_parts:
        ax.plot(projected_points[part, 0], projected_points[part, 1], marker='o')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()


# In[ ]:




