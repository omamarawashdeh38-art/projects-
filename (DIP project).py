#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pillow


# In[2]:


from PIL import Image 
from PIL import Image, ImageFilter


# In[7]:


# Read the images
lena_image = Image.open(r"C:\Users\user\Downloads\LenaGray.jpg")
pepper_image = Image.open(r"C:\Users\user\Downloads\PeppersGrey.jpg")


# combines the images based on the assignment requirement by looping through the pixels in both lena_image and pepper_image
#and combining them into a new image J. This code will create a new image J 
#where the top half is from lena_image and the bottom half is from pepper_image.
J = Image.new('L', (256, 256))
for y in range(256):
    for x in range(256):
        if y < 129:
            J.putpixel((x, y), lena_image.getpixel((x, y)))
        else:
            J.putpixel((x, y), pepper_image.getpixel((x, y)))

#rearrangement of the halves of the combined image J. It initializes a new image K, extracts the upper and lower halves from J,
#and then pastes them into K. By doing so, it rearranges the image to display
#the original lower half at the top and the original upper half at the bottom 
K = Image.new('L', (256, 256))
upper_half_J = J.crop((0, 0, 256, 128))
lower_half_J = J.crop((0, 128, 256, 256))
K.paste(lower_half_J, (0, 0))
K.paste(upper_half_J, (0, 128))

# Displaying the processed images
J.show()
K.show()


# In[8]:


left_half_lena = lena_image.crop((0, 0, 128, 256))
K.paste(left_half_lena, (128, 0))

right_half_pepper = pepper_image.crop((128, 0, 256, 256))
K.paste(right_half_pepper, (0, 0))
K.show()


# In[9]:


left_half_pepper = pepper_image.crop((0, 0, 128, 256))
K.paste(left_half_pepper, (128, 0))

right_half_lena = lena_image.crop((128, 0, 256, 256))
K.paste(right_half_lena, (0, 0))
K.show()


# In[5]:


# Load the noisy images
lena_noisy = Image.open(r"C:\Users\user\Downloads\LenaGrayNoisy.jpg").convert("L")  # Ensure grayscale
peppers_noisy = Image.open(r"C:\Users\user\Downloads\PeppersGreyNoisy.jpg").convert("L")  # Ensure grayscale

lena_noisy.show()
peppers_noisy.show()

# Function for Image Negative
def image_negative(image):
    return Image.eval(image, lambda x: 255 - x)

# Function for applying a 3x3 Median Filter
def median_filter(image):
    return image.filter(ImageFilter.MedianFilter(size=3))

# Applying Image Negative to the noisy images
lena_negative = image_negative(lena_noisy)
peppers_negative = image_negative(peppers_noisy)

# Applying Median Filter to the noisy images
lena_filtered = median_filter(lena_noisy)
peppers_filtered = median_filter(peppers_noisy)

# Display or save the resulting images
lena_negative.show()
peppers_negative.show()
lena_filtered.show()
peppers_filtered.show()

# Save the resulting images
lena_negative.save("LenaNegative.jpg")
peppers_negative.save("PeppersNegative.jpg")
lena_filtered.save("LenaFiltered.jpg")
peppers_filtered.save("PeppersFiltered.jpg")


# In[ ]:




