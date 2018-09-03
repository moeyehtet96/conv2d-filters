from PIL import Image
import numpy as np
# to add time taken feature later
#import time

def main():
    # ask the user for the input image
    img_dir = input("Select input image: ")
    
    # if image was found, open and convert to grayscale
    # if image cannot be found, exit the program
    try:
        img = Image.open(img_dir).convert("L")
    except IOError:
        print("Image Not Found")
        print("Exiting...")
        quit()
    
    # show the filter list
    print("Choose filter operation:\n" + "1. sharpen\n" + "2. blur\n" + "3. smooth\n" + 
        "4. Horizontal Sobel\n" + "5. Vertical Sobel\n" + "6. Lapalcian Edge Detector\n" + 
        "7. Lapalcian Edge Detector with diagonal\n" + "8. Emboss\n")
    
    # choose the desired filter and flip the kernel antidiagonally
    choice_num = int(input("Enter filter number: "))
    choice_filter = choose_filter(choice_num)
    flipped_filter = diagonal_flip(choice_filter)

    # perform the convolution
    img_filtered = conv2d(img,flipped_filter)

    # to add save to file feature later
    #imgout_name = input("Enter output image name with extension: ") # ask for file name to save the image

    # show both original and filtered image to the user
    img.show()
    img_filtered.show()

############################################################
# This function is for choosing the desired filter kernel. #
############################################################
def choose_filter(choice_num):
    # 2D Filter Kernels
    sharpen = np.array([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])

    blur = np.array([[1/16.0,1/8.0,1/16.0],
                     [1/8.0,1/4.0,1/8.0],
                     [1/16.0,1/8.0,1/16.0]])

    smooth = np.array([[1/9.0,1/9.0,1/9.0],
                       [1/9.0,1/9.0,1/9.0],
                       [1/9.0,1/9.0,1/9.0]])

    h_sobel = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])

    v_sobel = np.array([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]])

     # laplacian edge detector
    l_edge = np.array([[-1,-1,-1],
                       [-1, 4,-1],
                       [-1,-1,-1]])

    # laplacian edge detector with diagonals
    ld_edge = np.array([[-1,-1,-1],
                        [-1, 8,-1],
                        [-1,-1,-1]])

    emboss = np.array([[-2,-1,0],
                       [-1,1,1],
                       [0,1,2]])

    # assign the filter based on user choice
    if choice_num == 1:
        choice_mat = sharpen
    elif choice_num == 2:
        choice_mat = blur
    elif choice_num == 3:
        choice_mat = smooth
    elif choice_num == 4:
        choice_mat = h_sobel
    elif choice_num == 5:
        choice_mat = v_sobel
    elif choice_num == 6:
        choice_mat = l_edge
    elif choice_num == 7:
        choice_mat = ld_edge
    elif choice_num == 8:
        choice_mat = emboss

    return choice_mat

#############################################################
# This function is for flipping the kernel for convolution. #
#############################################################
def diagonal_flip(sel_kernel):
    kernel_hflip = np.fliplr(sel_kernel) # horizontal flip
    kernel_vflip = np.flipud(kernel_hflip) # vertical flip
    flipped_kernel = kernel_vflip.transpose() # transpose
    return flipped_kernel

################################################
# This function is for performing convolution. #
################################################
def conv2d(img, sel_filter):
    # convert image to a matrix with corresponding height and width
    img_mat = np.asarray(img.getdata(),dtype=float).reshape((img.size[1],img.size[0])) 

    # pad the image matrix with zeroes
    # so that the border pixels can be included in convolution
    img_mat_pad = np.pad(img_mat,(1,1),'constant')

    # determine the widths and heights of image and kernel matrices
    img_width, img_height = img_mat.shape[1], img_mat.shape[0]
    k_width, k_height = sel_filter.shape[1], sel_filter.shape[0]

    # matrix to store new pixel values
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
            # matrix to store product values
            product = np.zeros_like(sel_filter)

            for ky in range(k_height):
                for kx in range(k_width):
                    # multiply corresponding elements of image and kernel matrices
                    product[ky,kx] = sel_filter[ky,kx]*img_mat_pad[y+ky,x+kx]

            # calculate new pixel value
            img_mat_fil[y,x] = product.sum()

    # create image from the output matrix
    img_out = Image.fromarray(img_mat_fil)

    return img_out

if __name__ == "__main__":
    main()