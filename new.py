#!/usr/bin/env python
# coding: utf-8

# In[57]:


#1.Generate 100 random numbers and stored in an Array.  Search an element 100 in the array and record processing time of the machine.   

import random
import time

# Generate array of 100 random numbers
random.seed(111)
arr = [random.randint(1, 1000) for i in range(100)]

# Print the randomly generated array
print("Randomly generated array:", arr)

print()

# Record start time
start_time = time.perf_counter()

# Linear search for the number 100 in the array
found = False
for i in range(len(arr)):
    if arr[i] == 100:
        found = True
        break

# Record end time
end_time = time.perf_counter()

# Print the result and processing time
if found:
    print("Number 100 is found in the array.")
else:
    print("Number 100 is not found in the array.")
print("Processing time for searching: {:.10f} seconds".format(end_time - start_time))


# In[58]:


#2.Sort the same 100 generated numbers. Search an element, 100 in the sorted array. Record processing time. 
# Sort the array and record processing time
start_time_sort = time.perf_counter()
arr_sorted = sorted(arr)
end_time_sort = time.perf_counter()

# Print the sorted array and processing time
print("Sorted array: ", arr_sorted)
print()

sort_time = end_time_sort - start_time_sort
print("Processing time for sorting: {:.10f} seconds".format(sort_time))

# Search for the element 100 in the array and record processing time
start_time_search = time.perf_counter()
if 100 in arr_sorted:
    print("Element 100 found in the array")
else:
    print("Element 100 not found in the array")
end_time_search = time.perf_counter()

search_time = end_time_search - start_time_search
print()
print("Processing time for searching: {:.10f} seconds".format(search_time))

# Print the total processing time
print("Total processing time: {:.10f} seconds".format(sort_time + search_time))


# 3. Do experiments many times for searching different elements in both sorted array and unsorted array? Then write your observation based on processing time of each experiments. 
# 

# In[59]:


# Randomly select 10 numbers from the original array, and print them
random.seed(112)
random_numbers = random.sample(arr, 10)
print(random_numbers)
print()

# Initialize two arrays to store the processing time respectively 
unsorted_processing_times = []
sorted_processing_times = []

# Search for each selected number in the unsorted array and record the processing time
for num in random_numbers:
    start_time = time.perf_counter()

    found = False
    for i in range(len(arr)):
        if arr[i] == num:
            found = True
            break

    end_time = time.perf_counter()

    if found:
        print("Number", num, "found in the unsorted array.")
    else:
        print("Number", num, "not found in the unsorted array.")
        
    unsorted_processing_times.append(end_time - start_time)
    print("Processing time: {:.10f} seconds".format(end_time - start_time))
    print()

# Print all the processing time for searching in the unsorted array 
print()
print("Unsorted Processing times:", unsorted_processing_times)
print()

# Define a function to search for a number in the sorted array and record the processing time
def sorted_search(arr_sorted, x):
    start_time_sorted_elem = time.perf_counter()
    low = 0
    high = len(arr_sorted) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr_sorted[mid] < x:
            low = mid + 1
        elif arr_sorted[mid] > x:
            high = mid - 1
        else:
            end_time_sorted_elem = time.perf_counter()
            return mid, end_time_sorted_elem - start_time_sorted_elem
    end_time = time.perf_counter()  
    return -1, end_time_sorted_elem - start_time_sorted_elem  # if element is not found

# Apply the pre-defined function to each selected numbers
for num in random_numbers:
    index, processing_time = sorted_search(arr_sorted, num)
    if index != -1:
        print(f" {num} Found at index {index} in the sorted array, processing time: {processing_time:.10f} seconds.")
    else:
        print(f"{num} not found in {processing_time:.10f} seconds.")
    sorted_processing_times.append(processing_time)

# Print all the processing time for searching in the sorted array 
print()
print("sorted Processing times:", sorted_processing_times)
print()  

# Plot a scatterplot to compare the processing time for searching the same element in unsorted array and sorted array
import matplotlib.pyplot as plt

# Plot the processing times for the unsorted and sorted arrays as a scatter plot
plt.scatter(range(len(unsorted_processing_times)), unsorted_processing_times, label="Unsorted array")
plt.scatter(range(len(sorted_processing_times)), sorted_processing_times, label="Sorted array")
plt.title("Scatterplot of the Processing Times")
plt.xlabel("Experiment Number")
plt.ylabel("Time (seconds)")
plt.legend()
plt.show()


# The scatterplot illustrates the processing time for searching the same element in a sorted array is faster than that of an unsorted array in general.
# 
# The processing time for searching different elements in an unsorted array fluctuates significantly compared to that of a sorted array, which indicates the value and the location of the elements, as well as the size of the array, may affect the processing time.
# 
# Overall, sorting an array or not before searching for an element depends on the situation. Sorting an array before searching is generally an efficient strategy if we have to perform many searches for different elements because it can significantly reduce the search time. However, if we only need to perform a single search for a specific element or the size of the array is small,  we better consider whether sorting the array is worthwhile since it requires additional sorting time.

# In[60]:


##  假如不加时间 , Solution 2

# Randomly select 10 numbers from the original array
random.seed(112)
random_numbers = random.sample(arr, 10)
print(random_numbers)
print()

# Initialize two arrays to store the processing time respectively 
unsorted_processing_times = []
sorted_processing_times = []

# Search for each selected number in the unsorted array and record the processing time
for num in random_numbers:
    start_time_elem = time.perf_counter()

    found = False
    for i in range(len(arr)):
        if arr[i] == num:
            found = True
            break

    end_time_elem = time.perf_counter()

    if found:
        print("Number", num, "found in the array.")
    else:
        print("Number", num, "not found in the array.")
        
    unsorted_processing_times.append(end_time_elem - start_time_elem)
    print("Processing time: {:.10f} seconds".format(end_time_elem - start_time_elem))

print()
print("Unsorted Processing times:", unsorted_processing_times)
print()

# Search for the selected numbers in the sorted array and record the processing time
for num in random_numbers:
    start_time_search_elem = time.perf_counter()   
    
    found = False
    for i in range(len(arr_sorted)):
        if arr_sorted[i] == num:
            found = True
            break
            
    end_time_search_elem = time.perf_counter()
    
    if found:    
        print("Number {} found in the array".format(num))
    else:
        print("Number {} not found in the array".format(num))
        
    sorted_processing_times.append(end_time_search_elem - start_time_search_elem)
    print("Processing time: {:.10f} seconds".format(end_time_search_elem - start_time_search_elem))
    
print()
print("sorted Processing times:", sorted_processing_times)
print()  

# Plot a scatterplot to compare the processing time for searching the same element in unsorted array and sorted array
import matplotlib.pyplot as plt

plt.scatter(range(len(unsorted_processing_times)), unsorted_processing_times, label="Unsorted array")
plt.scatter(range(len(sorted_processing_times)), sorted_processing_times, label="Sorted array")
plt.title("Scatterplot of the Processing Times")
plt.xlabel("Search")
plt.ylabel("Time (seconds)")
plt.legend()
plt.show()


# 4.	Write any code for extracting dominant color in the images. Dominant color means, the color which share a greater number of pixels in the image.

# In[3]:


# Solution 1:
# Pre-install colorthief package by using the command 'pip install colorthief'
# Import libraries
# import colorthief
# from colorthief import ColorThief
# import matplotlib.pyplot as plt
# import colorsys

# # Loading the selected picture for extracing the color
# color = ColorThief('https://thumbs.dreamstime.com/b/beautiful-view-town-positano-antique-terrace-flowers-amalfi-coast-italy-balcony-92813755.jpg')

# # select the three most dorminant colors
# palette = color.get_palette(color_count = 3)
# plt.imshow([[palette[i] for i in range(3)]])
# plt.show()   

# # print the most dominant color and its index
# print('The most dominant color is: ', palette[0]) 
# print('The hexadecimal color code is', f"#{palette[0][0]:02x}{palette[0][1]:02x}{palette[0][2]:02x}")
# plt.imshow([[palette[0]]])
# plt.show()   


# In[3]:


from PIL import Image
import requests
from io import BytesIO

# Define a function to get the dominant color
def get_dominant_color(image):
    # Convert the image to RGB
    image = image.convert('RGB')

    # Get colors from the image
    pixels = image.getcolors(image.size[0]*image.size[1])

    # Sort colors by counting the number
    sorted_pixels = sorted(pixels, key=lambda t: t[0], reverse=True)

    # Get the most frequent color
    dominant_color = sorted_pixels[0][1]

    return dominant_color

# Load the selected image and apply the function
url = "https://thumbs.dreamstime.com/b/beautiful-view-town-positano-antique-terrace-flowers-amalfi-coast-italy-balcony-92813755.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
dominant_color = get_dominant_color(image)

# Print the dominant color
print('The dominant color of the image is:', dominant_color)
hex_color = '#{0:02x}{1:02x}{2:02x}'.format(*dominant_color)
print('The hexadecimal color code is', hex_color)

# Display a color patch with the dominant color
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.imshow(np.array([[dominant_color]]))
ax.set_title(hex_color)
plt.show()


# In[5]:


# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# import PIL
# %matplotlib inline

# def show_img_compar(img_1, img_2 ):
#     f, ax = plt.subplots(1, 2, figsize=(10,10))
#     ax[0].imshow(img_1)
#     ax[1].imshow(img_2)
#     ax[0].axis('off') #hide the axis
#     ax[1].axis('off')
#     f.tight_layout()
#     plt.show()
    
# img = cv.imread("https://thumbs.dreamstime.com/b/beautiful-view-town-positano-antique-terrace-flowers-amalfi-coast-italy-balcony-92813755.jpg")
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# dim = (500, 300)
# # resize image
# img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

# # print(img.shape)
# # img

# # print(img.reshape(-1,3).shape)
# # img.reshape(-1,3)

# # unique, counts = np.unique(img.reshape(-1,3), axis = 0, return_counts = True)
# # print(unique)
# # print(counts)

# img_temp = img.copy()
# unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
# img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[np.argmax(counts)]


# dominant_color = unique[np.argmax(counts)]
# print('The color with the greatest number of frequency is:', dominant_color)


# show_img_compar(img, img_temp)


# In[6]:


import urllib.request
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

def show_img_compar(img_1, img_2 ):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()

url = "https://thumbs.dreamstime.com/b/beautiful-view-town-positano-antique-terrace-flowers-amalfi-coast-italy-balcony-92813755.jpg"
urllib.request.urlretrieve(url, "image.jpg")
img = cv.imread("image.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

dim = (500, 300)
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

img_temp = img.copy()
unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[np.argmax(counts)]

dominant_color = unique[np.argmax(counts)]
print('The color with the greatest number of frequency is:', dominant_color)

show_img_compar(img, img_temp)

