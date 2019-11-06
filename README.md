# HiC_data_generator

The input files are hiccups looplist and bedpe formatted files. The generator splits the input that into batches and uses pybedtools to extract reads overlapping the loops.  
Generates batches of data of shape:  
X is a matrix of shape (batch_size) * (radius/binsize*2+1) * (radius/binsize*2+1)  
 
The main purpose of the class is to use the augmentator. You can specify the augmentation ratio, which will generate downsampled windows of the same size, proportional to the observed HiC count. If random_shifts is disabled, the hiccups cluster center falls at the center of the matrices in X. If random_shifts is set to True, the cluster center randomly moves in the window, and the matrix is completed by taking the first quantile of the diagonals.  

I will have to provide a better documentation later.
