from PIL import Image
import numpy as np

def main():
	img = Image.open("wolf.jpg").convert("L") # open image, convert to grayscale
	img_mat = np.asarray(img.getdata(),dtype=float).reshape((img.size[1],img.size[0])) # turn image into matrix
	T_calc = findThreshold(img_mat)
	img_segmented_mat = doSegment(img_mat,T_calc)
	img_segmented =  Image.fromarray(img_segmented_mat)
	img.show()
	img_segmented.show()

def findThreshold(img_mat):
	hist_bin = np.arange(256) # define histogram bins
	hist = np.histogram(img_mat,bins=256,range=(0,256)) # find histogram 
	hist_val = hist[0] # extract histogram values

	bc_var = np.zeros_like(hist_bin)
	bc_var[0] = 0

	for T in range(1,256):
		hist_val_back = hist_val[0:T] # histogram values below threshold
		hist_bin_back = hist_bin[0:T] # bin values below threshold
		hist_val_fore = hist_val[T:256] # histogram values above threshold
		hist_bin_fore = hist_bin[T:256] # bin values above threshold

		back_sum = np.sum(hist_val_back,dtype=np.float_) # sum of background values
		fore_sum = np.sum(hist_val_fore,dtype=np.float_) # sum of foreground values
		pix_total = np.sum(hist_val,dtype=np.float_) # total pixel number

		# weights for background and foreground
		w_b = back_sum/pix_total
		w_f = fore_sum/pix_total

		# sum of products of histogram value and intensity value for background and foreground
		prodB_sum = float(np.dot(hist_val_back,hist_bin_back))
		prodF_sum = float(np.dot(hist_val_fore,hist_bin_fore))

		if back_sum == 0:
			mean_b = 0
		else:
			mean_b = prodB_sum/back_sum

		if fore_sum == 0:
			mean_f = 0
		else:
			mean_f = prodF_sum/fore_sum

		# Between class variance
		bc_var[T] = w_b*w_f*((mean_b-mean_f)**2)

	threshold_calc = bc_var.argmax()

	return threshold_calc

def doSegment(img_mat,calc_threshold):
	img_seg_mat = np.zeros_like(img_mat)

	img_width, img_height = img_mat.shape[1], img_mat.shape[0]

	for y in range(img_height):
		for x in range(img_width):
			if img_mat[y,x] >= calc_threshold:
				img_seg_mat[y,x] = 255
			else:
				img_seg_mat[y,x] = 0

	return img_seg_mat

if __name__ == "__main__":
    main()

