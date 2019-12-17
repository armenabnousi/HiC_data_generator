#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tensorflow.keras.utils import Sequence
import pandas as pd
from subprocess import check_output, check_call
import subprocess
from pybedtools import BedTool
import random
from scipy import sparse
import glob
import os


class DataGenerator(Sequence):
    def __init__(self, hiccups_filename, hic_prefix, hic_suffix, binsize, radius, 
                 augmentation_rate = 5, downsample_min = 1, random_shifts = True,
                 chromosome_count = 22, to_fit=True, batch_size=32,
                 n_channels=1, shuffle=True, in_memory = False):
        self.hiccups_filename = hiccups_filename
        self.interactions_filename = self.hiccups_filename
        self.hic_prefix = hic_prefix
        self.hic_suffix = hic_suffix
        self.binsize = binsize
        self.radius = int(np.ceil(radius // self.binsize) * self.binsize)
        self.augmentation_rate = augmentation_rate
        self.to_fit = to_fit
        self.random_shifts = random_shifts
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.chromosome_count = chromosome_count
        self.shuffle = shuffle
        self.in_memory = in_memory
        self.batch_indices = None
        #self.convert_hiccups(self.interactions_filename, self.hiccups_filename)
        self.interactions_count = int(check_output(["wc", "-l", self.interactions_filename]).split()[0])
        self.interactions = pd.read_csv(self.interactions_filename, sep = "\t", header = None)
        self.pre_augment_batch_size = int(np.ceil(self.batch_size / self.augmentation_rate))
        self.downsample_min = downsample_min
        self.dim = int(self.radius / self.binsize) * 2 + 1
        print(self.interactions_count, self.batch_size, self.pre_augment_batch_size, self.dim)
        if (self.in_memory):
            self.data = self._extract_bedpe_intersections(self.interactions_filename)
        self.on_epoch_begin()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(self.interactions_count / self.pre_augment_batch_size))

    def __getitem__(self, index):
        
        #subset interactions for the batch
        #interactions = self.interactions.iloc[batch_indices,:]
        if not self.in_memory:
            interactions_bedpe = BedTool(self.interactions_filename + "_mbatch" + str(index))
        else:
            interactions_bedpe = self.interactions.iloc[self.batch_indices[index],:]

        # Generate data
        X, y = self._generate_data(interactions_bedpe, self.to_fit)

        if self.to_fit:
            return X, y
        else:
            return X

    def on_epoch_begin(self):        
        num_batches = int(np.ceil(self.interactions_count / self.pre_augment_batch_size))
        #print(num_batches)
        d = pd.read_csv(self.interactions_filename, sep = "\t", header = None)
        d['class'] = 'p'
        d.loc[d.loc[:,7] < 0,'class'] = 'n'
        d_pos = d[d['class'] == 'p']
        d_neg = d[d['class'] == 'n']
        #print(d_pos.shape)
        #print(d_neg.shape)
        pre_aug_pos_count = int(np.ceil(float(d_pos.shape[0])/d.shape[0] * self.pre_augment_batch_size))
        pre_aug_neg_count = self.pre_augment_batch_size - pre_aug_pos_count
        #print("sizes:")
        #print(pre_aug_pos_count)
        #print(pre_aug_neg_count)
        if not self.in_memory:
            if self.shuffle == True:
                d_pos = d_pos.sample(frac=1).reset_index(drop=True)
                d_neg = d_neg.sample(frac=1).reset_index(drop=True)
            for old_file in glob.glob("data_splits/train.datagen_mbatch*"):
                os.remove(old_file)
            for batch_num in range(num_batches):
                batch_pos = d_pos.iloc[(batch_num*pre_aug_pos_count):((batch_num+1)*pre_aug_pos_count),:]
                batch_neg = d_neg.iloc[(batch_num*pre_aug_neg_count):((batch_num+1)*pre_aug_neg_count),:]
                batch_all = pd.concat([batch_pos, batch_neg], axis = 0)
                batch_all.drop('class', axis = 1, inplace = True)
                batch_all.sample(frac=1).reset_index(drop = True)
                batch_all.to_csv(self.interactions_filename + "_mbatch" + str(batch_num), sep = "\t", header = None, index = False)
        else:
            pos_indices = list(d_pos.index.values)
            neg_indices = list(d_neg.index.values)
            if self.shuffle == True:
                random.shuffle(pos_indices)
                random.shuffle(neg_indices)
            batch_indices = {}
            for batch_num in range(num_batches):
                batch_pos_indices = pos_indices[(batch_num*pre_aug_pos_count):((batch_num+1)*pre_aug_pos_count)]
                batch_neg_indices = neg_indices[(batch_num*pre_aug_neg_count):((batch_num+1)*pre_aug_neg_count)]
                batch_indices[batch_num] = batch_pos_indices + batch_neg_indices
            self.batch_indices = batch_indices

    def _generate_data(self, interactions_bedpe, to_fit):
        # Generate data
        if not self.in_memory:
            df = self._extract_bedpe_intersections(interactions_bedpe)
        else:
            interactions_bedpe = interactions_bedpe.iloc[:, [0,1,2,3,4,5]]
            merge_columns = ["chr1", "cluster_x1", "cluster_x2", "chr2", "cluster_y1", "cluster_y2"]
            interactions_bedpe.columns = merge_columns
            #print(self.data.info())
            #print(interactions_bedpe.info())
            df = self.data.merge(interactions_bedpe)
            #print(interactions_bedpe.shape)
            #print(self.data.shape)
            #print(df.shape)
        #print("now grouping")
        #print(df.head())
        #print(df.tail())
        print(df.shape)
        df = df.groupby("cluster_name").apply(DataGenerator._make_sparse_matrix, binsize = self.binsize, radius = self.radius).reset_index()
        #df['cluster_number'] = df.groupby(["cluster_name"]).ngroup()
        if self.to_fit:
            if self.augmentation_rate > 1:
                df = df.groupby(["cluster_name"]).apply(self._augment_matrix, self.augmentation_rate, self.downsample_min)
                df.reset_index(drop = True, inplace = True)
            if self.random_shifts:
                df = df.groupby('cluster_name').apply(self._generate_random_moves, self.radius, self.binsize).reset_index()
            df['cluster_number'] = df.groupby(["cluster_name"]).ngroup()
            y = self._generate_y(df)
            X = self._generate_all_full_matrices(df, self.random_shifts)            
            
        else:
            y = 0
            X = self._generate_all_full_matrices(df)
            #cluster_count = len(df['cluster_name'].drop_duplicates())
            #X = df
            #X = roll_matrices
            ####****separate the .datagen generation because it needs to be done before so the negative samples can be added.
        X = X.reshape(X.shape + (1,)) 
        return X, y
    
    def _generate_y(self, df):
        ydata = df.groupby('cluster_name').apply(self._get_ydata, self.radius, self.binsize, self.random_shifts)
        ydata = np.array([*ydata])
        return ydata
       
    @staticmethod
    def _get_ydata(df, radius, binsize, random_shifts):
        #print(df.head())
        radius_unit = radius / binsize
        cluster_radius = df.iloc[0]['cluster_radius']
        cluster_number = df.iloc[0]['cluster_number']
        if random_shifts:
            x_move, y_move = df.iloc[0]['x_move'], df.iloc[0]['y_move']
        else:
            x_move, y_move = 0, 0
        if cluster_radius >= 0:
            is_true = 1
            pos_x, pos_y = radius_unit + y_move, radius_unit + x_move
        else:
            is_true = pos_x = pos_y = 0
            pos_x, pos_y = x_move, y_move
        return is_true, cluster_radius/radius_unit, pos_x/(2*radius_unit+1), pos_y/(2*radius_unit+1) #, x_move, y_move, cluster_number
        
            

    def _extract_bedpe_intersections(self, interactions_bedpe):
        df = pd.DataFrame()
        for i in range(1, self.chromosome_count + 1):
            chr_bedpe = BedTool(self.hic_prefix + str(i) + self.hic_suffix)
            #print(vars(interactions_bedpe))
            #print(vars(chr_bedpe))
            intersection = chr_bedpe.pairtopair(interactions_bedpe, type = "both")
            df_chr = pd.read_table(intersection.fn, names = ['chr1', 'x1', 'x2', 'chr2', 
                                                              'y1', 'y2', 'ignore1', 'count', 
                                                              'ignore2', 'ignore3', 'ignore4', 
                                                              'cluster_x1', 'cluster_x2', 'ignore5',
                                                             'cluster_y1', 'cluster_y2', 'cluster_name', 
                                                              'cluster_radius', 'ignore6', 'ignore7'])
            df_chr.drop_duplicates(inplace = True)
            df = pd.concat([df, df_chr], axis = 0)
            #print("chr", i)
            #print(df.head())
            #print(df.tail())
            print(df_chr.shape)
        df['cluster_name'] = df.groupby(['chr1', 'chr2', 'cluster_x1', 'cluster_x2', 'cluster_y1', 'cluster_y2']).ngroup()
        df.reset_index(drop = True, inplace = True)
        #print("final intersected:")
        #print(df.head())
        #print(df.tail())
        return df
    
    def _generate_all_full_matrices(self, sparse_matrices, random_shifts = False):
        cluster_names = sparse_matrices['cluster_name'].drop_duplicates()
        labels = {}
        '''
        cntr = 0
        for name in cluster_names:
            labels[name] = cntr
            cntr += 1
        '''
        cluster_count = len(cluster_names)
        n = np.zeros((cluster_count, self.dim, self.dim))
        #print(n.shape)
        sparse_matrices.groupby('cluster_name', as_index = False).apply(self._make_full_matrix, dim = self.dim, mat = n, labels = labels, random_shifts = random_shifts)
        return n
    
    @staticmethod
    def _make_full_matrix(d, dim, mat, labels, random_shifts):
        cluster_name = d.iloc[0]['cluster_name']
        #first_dim = labels[cluster_name]
        first_dim = d.iloc[0]['cluster_number']
        max_i_j = mat.shape[1]
            
        submat = sparse.csc_matrix((d['count'], (d['i'], d['j'])), shape = (mat.shape[1], mat.shape[2])).todense()
        if random_shifts:
            #print(d.head())
            x_move = d.iloc[0]['x_move']
            y_move = d.iloc[0]['y_move']
            submat = DataGenerator._move_with_imputation_matrix(submat, x_move, y_move)
            '''
            
            d['i'] = d['i'] + x_move
            d['j'] = d['j'] + y_move
            '''

        mat[first_dim] = submat
    
    def convert_hiccups(self, converted, hiccups):
        d = pd.read_csv(hiccups, sep = "\t")
        d = d[d['chr1'] == d['chr2']]
        d = d[['chr1', 'centroid1', 'centroid2', 'radius', 'fdr_donut']].drop_duplicates()
        #print(vars(self))
        centroid_x = d['centroid1'] // self.binsize * self.binsize
        centroid_y = d['centroid2'] // self.binsize * self.binsize
        '''
        d['x1'] = d['centroid1'] - self.radius #- d['radius']
        d['x2'] = d['centroid1'] + self.radius #+ d['radius']
        d['y1'] = d['centroid2'] - self.radius #- d['radius']
        d['y2'] = d['centroid2'] + self.radius #+ d['radius']
        '''
        d['x1'] = centroid_x - self.radius
        d['x2'] = centroid_x + self.radius + self.binsize
        d['y1'] = centroid_y - self.radius
        d['y2'] = centroid_y + self.radius + self.binsize
        d['radius'] = np.floor(d['radius'] / self.binsize)
        d['strand1'] = '+'
        d['strand2'] = '+'
        d['name'] = list(range(d.shape[0]))
        d = d[['chr1', 'x1', 'x2', 'chr1', 'y1', 'y2', 'name', 'radius', 'strand1', 'strand2']]
        d.to_csv(converted, sep = "\t", index = False, header = False)
        
    @staticmethod
    def _make_sparse_matrix(df, binsize, radius):
        #print("inmakesparse")
        #print(df.head())
        result = pd.DataFrame({'count' : df['count'], 'cluster_name' : df.iloc[0]['cluster_name']})
        #print(df.head())
        center_x = np.floor(((df.iloc[0]['cluster_x1'] + df.iloc[0]['cluster_x2']) / 2) / binsize) * binsize
        center_y = np.floor(((df.iloc[0]['cluster_y1'] + df.iloc[0]['cluster_y2']) / 2) / binsize) * binsize
        result['i'] = np.int64((df['x1'] - center_x + radius) / binsize)
        result['j'] = np.int64((df['y1'] - center_y + radius) / binsize)
        result['center_x'] = center_x
        result['center_y'] = center_y
        #result['count'] = df['count'] ####SOMETHING IS WRONG WITH THE COUNT IT BECOMES NAN
        result['cluster_radius'] = df.iloc[0]['cluster_radius']
        #print(df.head())
        return result
    
    @staticmethod
    def _sample_matrix(d, sampling_rate, name_suffix):
        n = int(np.floor(sampling_rate * d['count'].sum()))
        sample = d.sample(n, replace = True, weights = d['count'])
        sample['count'] = sample.groupby(['i', 'j']).transform(np.size)
        sample.drop_duplicates(inplace = True)
        sample['cluster_name'] = sample['cluster_name'].astype(str) + ("_" + str(name_suffix))
        return sample

    @staticmethod
    def _augment_matrix(d, rate, downsample_min):
        sampling_rates = np.random.uniform( 1./downsample_min, 1, rate)
        samples = [DataGenerator._sample_matrix(d, i, j) for i, j in zip(sampling_rates, range(rate))]
        samples = pd.concat(samples, axis = 0, ignore_index = True)
        return samples
    
    @staticmethod
    def _generate_random_moves(d, radius, binsize):
        d['diagonal'] = 0
        cluster_radius = d.iloc[0]['cluster_radius']
        max_move = (radius - cluster_radius) // binsize
        x_move, y_move = np.random.randint(-max_move, high = max_move + 1, size = 2)
        
        #check if its triangular; if it is make the random moves on the diagonal
        candidate_triangular_i = d[d['j'] == 0]['i'].max()
        if candidate_triangular_i < (radius/binsize) * 2:
            candidate_triangular_j = d[d['i'] == (radius/binsize) * 2]['j'].min()
            if candidate_triangular_i == candidate_triangular_j:
                y_move = x_move
                d['diagonal'] = 1
        d['x_move'] = x_move
        d['y_move'] = y_move
        return d
    
    @staticmethod
    def _impute_diag(mat, direction):
        m,n = mat.shape
        aux = np.full((m,2*n+1), np.nan)
        np.copyto(aux[:,:n],mat)
        aux = aux.ravel()[:-m].reshape(m, 2*n)
        if direction == 'top':
            aux = aux[:,-m-1:-1]
        elif direction == 'bottom':
            aux = aux[:,:m]
        return aux

    @staticmethod
    def _move_with_imputation_matrix(mat, x_move, y_move):
        if x_move == 0 and y_move == 0:
            return mat
        quants_upper = [np.floor(np.quantile(mat.diagonal(offset = i), q = 0.25)) for i in range(mat.shape[1])]
        quants_lower = [np.floor(np.quantile(mat.diagonal(offset = -i), q = 0.25)) for i in range(mat.shape[0])]
        imputes_bl = DataGenerator._impute_diag(np.tile(np.flip(quants_lower), (len(quants_lower),1)), 'bottom') #for bottom and left we use the same matrix
        imputes_tr = DataGenerator._impute_diag(np.tile(quants_upper, (len(quants_upper),1)), 'top') #for top and right we use the same matrix
        temp = np.full(mat.shape, np.nan)

        if (x_move >= 0 and y_move >= 0):
            end_x = end_y = max(temp.shape)
            if x_move == 0:
                base = mat[:, :-y_move]
                end_x = -1
            elif y_move == 0:
                base = mat[:-x_move, :]
                end_y = -1
            else:
                base = mat[:(-x_move),:(-y_move)]
            temp[-base.shape[0]:, -base.shape[1]:] = base
            imputes_bl2 = DataGenerator._impute_diag(np.tile(quants_lower, (len(quants_lower),1)), 'bottom') #for bottom and left we use the same matrix
            imputes_tr2 = DataGenerator._impute_diag(np.tile(np.flip(quants_upper), (len(quants_upper),1)), 'top') #for top and right we use the same matrix
            np.fill_diagonal(imputes_tr2, np.nan)
            complement = np.transpose(np.nansum(np.dstack((imputes_bl2, imputes_tr2)),2))
            if x_move > 0 and y_move > 0: 
                temp[:-base.shape[0], :-base.shape[1]] = complement[-(temp.shape[0]-base.shape[0]):, -(temp.shape[1] - base.shape[1]):]
            if y_move != 0:    
                temp[-base.shape[0]:end_x, :(temp.shape[1] - base.shape[1])] = imputes_bl[1:base.shape[0]+1, -(temp.shape[1] - base.shape[1]):] 
            #print((temp.shape[0] - base.shape[0]))
            #print(-base.shape[1])
            if x_move != 0:
                temp[:(temp.shape[0] - base.shape[0]) , -base.shape[1]:end_y] = imputes_tr[-(temp.shape[0]-base.shape[0]): ,1:base.shape[1]+1]

        elif (x_move <= 0 and y_move <= 0):
            base = mat[abs(x_move):, abs(y_move):]
            begin_x = begin_y = 0
            if x_move == 0:
                temp[:, :y_move] = base
                begin_x = 1
            elif y_move == 0:
                temp[:x_move, :] = base
                begin_y = 1
            else:
                temp[:x_move, :y_move] = base
            imputes_bl2 = DataGenerator._impute_diag(np.tile(quants_lower, (len(quants_lower),1)), 'bottom') #for bottom and left we use the same matrix
            imputes_tr2 = DataGenerator._impute_diag(np.tile(np.flip(quants_upper), (len(quants_upper),1)), 'top') #for top and right we use the same matrix
            np.fill_diagonal(imputes_tr2, np.nan)
            complement = np.transpose(np.nansum(np.dstack((imputes_bl2, imputes_tr2)),2))
            temp[base.shape[0]:, base.shape[1]:] = complement[:(temp.shape[0]-base.shape[0]), :(temp.shape[1]-base.shape[1])]
            temp[base.shape[0]:, begin_y:base.shape[1]] = imputes_bl[:(temp.shape[0]-base.shape[0]), -base.shape[1]-1:-1]
            temp[begin_x:base.shape[0], base.shape[1]:] = imputes_tr[-base.shape[0]-1:-1, :(temp.shape[1]-base.shape[1])]
        elif (x_move > 0 and y_move <= 0):
            if y_move == 0:
                base = mat[x_move:, :]
            else:
                base = mat[x_move:, :y_move]
            temp[:-x_move, abs(y_move):] = base
            temp[(base.shape[0]):, (temp.shape[1] - base.shape[1]+1):] = imputes_bl[:(temp.shape[0]-base.shape[0]), :(base.shape[1]-1)] #impute bottom triangle
            temp[:base.shape[0]-1, :(temp.shape[1] - base.shape[1])] = imputes_bl[-base.shape[0]+1:, -(temp.shape[1]-base.shape[1]):] #impute left triangle 
        else: #x_move < 0 and y_move > 0
            if x_move == 0:
                base = mat[:, y_move:]
            else:
                base = mat[:x_move, y_move:]
            temp[-base.shape[0]:, :base.shape[1]] = base
            temp[(temp.shape[0]-base.shape[0]+1):, base.shape[1]:] = imputes_tr[:(base.shape[0]-1), :(temp.shape[1]-base.shape[1])] #impute right triangle
            temp[:-base.shape[0], :base.shape[1]-1] = imputes_tr[-(temp.shape[0]-base.shape[0]):, -base.shape[1]+1:] #impute top triangle
        temp1 = temp[:, :int(np.ceil(temp.shape[1])/2)]
        temp2 = temp[:, int(np.ceil(temp.shape[1])/2):]
        temp1[np.isnan(temp1)] = 0 #np.nanmax(temp) #zero will simulate crossing main diagonal in upper triangular
        temp2[np.isnan(temp2)]= np.nanmin(temp)
        temp = np.concatenate([temp1, temp2], axis = 1)
        return temp

