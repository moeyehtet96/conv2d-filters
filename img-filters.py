from PIL import Image
import numpy as np
import time

# Load the image and convert to grayscale and show it
# Then, convert it to a matrix with corresponding height and width
# Then, pad the matrix with zeroes.
# Reason for padding with zero is so that we can also calculate the border pixels
img = Image.open("/home/moeyehtet96/Downloads/wolf.jpg").convert("L")
img.show("Original Image")
img_mat = np.asarray(img.getdata(),dtype=float).reshape((img.size[1],img.size[0]))
img_mat_pad = np.pad(img_mat,(1,1),'constant')

# define the image filter kernel
kernel = np.array([[-1,-2,-1],
                   [0,0,0],
                   [1,2,1]])

# define different kernels
sharpen = np.array([[0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]])

blur = np.array([[0.0625,0.125,0.0625],
                 [0.125,0.25,0.125],
                 [0.0625,0.125,0.0625]])



# find width and height for padded image matrix and kernel matrix for the for loop
img_width, img_height = img_mat.shape[1], img_mat.shape[0]
k_width, k_height = kernel.shape[1], kernel.shape[0]

# create a zero matrix to store output pixels
img_mat_fil = np.zeros_like(img_mat)

# This part is for convolution.
# It iterates over the padded image matrix and calculates the new pixel value.
# For example,
# padded_matrix = [[0,0,0,0,0],
#                  [0,a,b,c,0],
#                  [0,d,e,f,0],
#                  [0,g,h,i,0],
#                  [0,0,0,0,0]]
# The code iterates from column #0 to column #2 and from row #0 to row #2
# Take for example, the iteration for row #0 and column #1.
# The 9 elements we are interested in are [[0,0,0],
#                                          [a,b,c],
#                                          [d,e,f]]
# Each of these elements are multiplied with the corresponding kernel matrix elements.
# Then, the sum of these products and the sum of the kernel elements are taken.
# The new pixel value is calculated by dividing the product sum by kernel sum.
# The new pixel value is put in the row #0, column #1 position in the filtered image.
for y in range(img_height):
    for x in range(img_width):
        product = np.zeros_like(kernel) # create an array to store products

        # For the pixel position in above for loops,
        # multiply corresponding pixels with the kernel
        for ky in range(k_height):
            for kx in range(k_width):
                product[ky,kx] = kernel[ky,kx]*img_mat_pad[y+ky,x+kx]

        # calculate new pixel values from sum of the product matrix and sum of kernel matrix
        img_mat_fil[y,x] = product.sum()#/kernel.sum()

# create an image from the matrix used to hold the new pixel values
img_fil = Image.fromarray(img_mat_fil)
img_fil.show("Filtered Image") # show the filtered image