# HiC_data_generator
## A tf.keras data generator for HiC data
The input files are hiccups looplist and bedpe formatted files. The generator splits the input that into batches and uses pybedtools to extract reads overlapping the loops.  
Generates batches of data of shape:  
X is a matrix of shape (batch_size) * (radius/binsize*2+1) * (radius/binsize*2+1)  
y is a 2d matrix, each row pertains to one input point from the HICCUPS looplist (and the augmentations if you have used it). Indexes in y are:  
index0: whether it is a interaction or not (0/1). For the non-interactions set radius to -1 in the input hiccups file.  
index1: radius of the interaction cluster detected by HICCUPS  
index2,3: if the row indicates an interaction, second and third indices indicate the center of the interaction cluster as defined by HICCUPS (if it is not an interaction, disregard this indices).  
indices4+: disregard these indices, I used them for debugging but decided to keep them for now.  
 
The main purpose of the class is to use the augmentator. You can specify the augmentation ratio, which will generate downsampled windows of the same size, proportional to the observed HiC count. If random_shifts is disabled, the hiccups cluster center falls at the center of the matrices in X. If random_shifts is set to True, the cluster center randomly moves in the window, and the matrix is completed by taking the first quantile of the diagonals.  

I will have to provide a better documentation later.
