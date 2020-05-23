import os
import time
import numpy as np
import xarray as xr
import tensorflow as tf
 

class Normalizer(object):
    def __init__(self, data_dir, norm_fn, fsub, fdiv, tmult):
        self.data_dir = data_dir
        self.fsub     = fsub
        self.fdiv     = fdiv
        self.tmult    = tmult
        
        norm_ds = xr.open_dataset(
            os.path.join(data_dir, norm_fn)
        )
        
        self.feature_norms = [
            norm_ds[fsub][:].data,
            norm_ds[fdiv][:].data
        ]
        
        self.target_norms = [0., norm_ds[tmult][:].data]
        
    def transform(self, x, y):
        x = (x - self.feature_norms[0]) / self.feature_norms[1]
        y = (y - self.target_norms[0]) * self.target_norms[1]
        
        return x, y
    
    
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data_dir, feature_fn, target_fn, transform, batch_size=1024, shuffle=True, val=False):
        
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.transform  = transform
        
        self.val = val
        # Open datasets
        self.features_ds = xr.open_dataset(
            os.path.join(data_dir, feature_fn)
        )
        self.targets_ds  = xr.open_dataset(
            os.path.join(data_dir, target_fn)
        )

        # Compute number of samples and batches
        self.n_samples = self.features_ds['features'].shape[0]
        if val: self.n_samples /= 3
        self.n_batches = int(np.floor(self.n_samples) / self.batch_size)  
        
        self.start_time = time.time()
        
    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        # Compute start and end indices for batch
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        # Grab batch from data
        X = self.features_ds['features'][start_idx:end_idx].data
        Y = self.targets_ds['targets'][start_idx:end_idx].data

        # Normalize
        X, Y = self.transform.transform(X, Y)
      
        return X, Y

    def on_epoch_end(self):
        self.indices = np.arange(self.n_batches)
        if self.shuffle: np.random.shuffle(self.indices)
        if self.val:
            print((time.time() - self.start_time) / 60., flush=True)
        
            self.start_time = time.time()