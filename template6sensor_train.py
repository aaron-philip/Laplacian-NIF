from sys import path
path.append("/mnt/scratch/philipaa/tddft-emulation/nif")
import os
import subprocess
from scipy.io import FortranFile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import nif
import numpy as np
import time
import logging
from contextlib import nullcontext
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from nif.optimizers import centralized_gradients_for_optimizer
import h5py
import random
import math
import keras
from io_funcs import read_tdd, gen_sample, generator, format_input_sensor6

""" Updates the initial learning rate based on current epoch number"""
def scheduler(epoch, lr):
    if epoch < nepoch/5:
            return learn
    elif epoch < nepoch/2.5:
            return learn/5
    elif epoch < nepoch/1.25:
            return learn/10
    else:
            return learn/50

""" Creates training log, periodically checkpoints the model, plots predictions vs truth"""
# custom callback 
class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_begin_time = time.time()
        self.history_loss = []
        logging.basicConfig(filename='{}/log'.format(direc), level=logging.INFO, format='%(message)s')

    def on_epoch_begin(self, epoch, logs=None):
        self.ts = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % display_epoch == 0:
            tnow = time.time()
            te = tnow - self.ts
            logging.info("Epoch {:6d}: avg.loss pe = {:4.3e}, {:d} points/sec, time elapsed = {:4.3f} hours".format(
                epoch, logs['loss'], int(batch_size / te), (tnow - self.train_begin_time) / 3600.0))
            self.history_loss.append(logs['loss'])

        if epoch % checkpt_epoch == 0 or epoch == nepoch - 1:
            print('save checkpoint epoch: %d...' % epoch)
            self.model.save_weights("{}/saved_weights/ckpt-{}/ckpt".format(direc,epoch))
    
        if epoch % print_figure_epoch == 0:
            plt.figure()
            plt.semilogy(self.history_loss)
            plt.xlabel('epoch: per {} epochs'.format(print_figure_epoch))
            plt.ylabel('MSE loss')
            plt.savefig('{}/loss.png'.format(direc, epoch))
            plt.close()

            t_0 = 500
            tdd_dir = "/mnt/home/philipaa/tddft-emulation/Sky3D-1.1/Code/test/dynamicCa40/"
            filename = '{:06d}.tdd'.format(t_0)
            rho0, current0, _ = read_tdd(filename, tdd_dir)
            
            factors_path_x = "../sensor6normalization_full_factors_x.npy"
            factors_path_y = "../sensor6normalization_full_factors_y.npy"
            In = format_input_sensor6(rho0, current0, factors_path_x)
            In = In.reshape(-1, 27)
            pred = self.model.predict_on_batch(In)
            
            y_shift, y_scale = np.load(factors_path_y)    
            pred = pred.reshape(96,96,8)*y_scale + y_shift
            
            print("Printing Prediction Figure Epoch ", epoch)
            fig,axs=plt.subplots(1,3,figsize=(16,8))
            im1 = axs[0].imshow(pred[:,:,4], origin='lower',vmax=max(pred[:,:,4].max(),rho0[:,:,11].max()), vmin=min(pred[:,:,4].min(),rho0[:,:,11].min()))
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            axs[0].set_title("Predicted Density: Z = 12")
            plt.colorbar(im1,ax=axs[0])

            im2 = axs[1].imshow(rho0[:,:,11], origin ='lower',vmax=max(pred[:,:,4].max(),rho0[:,:,11].max()), vmin=min(pred[:,:,4].min(),rho0[:,:,11].min())) 
            axs[1].set_xlabel('X')
            axs[1].set_ylabel('Y')
            axs[1].set_title("True Density: Z = 12")
            plt.colorbar(im2,ax=axs[1])

            im3 = axs[2].imshow(rho0[:,:,11]-pred[::4,::4,4], origin ='lower') 
            axs[2].set_xlabel('X')
            axs[2].set_ylabel('Y')
            axs[2].set_title("Difference: True - Predicted")
            plt.colorbar(im3,ax=axs[2])

            plt.savefig('{}/vis_multiscale_{}.png'.format(direc, epoch))
            plt.close()
            
if __name__ == '__main__':
    
    N_S_Units = 0
    N_S_Layers = 0
    N_P_Units = 0
    N_P_Layers = 0
    Rank_Linear = 0
    extra = "6sensor"

    direc = "../Models/{}-{}-{}-{}-{}-{}".format(extra, N_S_Units, N_S_Layers, N_P_Units, N_P_Layers, Rank_Linear)
    
    os.system("rm -r %s" % direc)
    os.system("mkdir %s" % direc)
    enable_multi_gpu = False
    enable_mixed_precision = False

    # mixed precision?
    if enable_mixed_precision:
        mixed_policy = "mixed_float16"
        # we might need this for `model.fit` to automatically do loss scaling
        policy = nif.mixed_precision.Policy(mixed_policy)
        nif.mixed_precision.set_global_policy(policy)
    else:
        mixed_policy = 'float32'


    cfg_shape_net = {
        "use_resblock":True,
        "connectivity": 'last_layer',
        "input_dim": 3,
        "output_dim": 8,
        "units": N_S_Units,
        "nlayers": N_S_Layers,
        "activation": 'swish',
        "weight_init_factor": 0.01,
        "omega_0":30.0
    }
    cfg_parameter_net = {
        "use_resblock":True,
        "input_dim": 24,
        "latent_dim": Rank_Linear,
        "units": N_P_Units,
        "nlayers": N_P_Layers,
        "activation": 'swish',
    }


    nepoch = 200
    learn = 1e-4
    batch_size = 1600
    checkpt_epoch = nepoch/10
    display_epoch = checkpt_epoch
    print_figure_epoch = 10#display_epoch

    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    TerminateOnNaN = tf.keras.callbacks.TerminateOnNaN()
    callbacks = [TerminateOnNaN, LossAndErrorPrintingCallback(), scheduler_callback]

    cm = tf.distribute.MirroredStrategy().scope() if enable_multi_gpu else nullcontext()
    with cm:
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.MeanSquaredError()
        model_ori = nif.NIFMultiScaleLastLayerParameterized(cfg_shape_net, cfg_parameter_net, mixed_policy) # LastLayerParameterized
        model_opt = model_ori.build()
        model_opt.compile(optimizer, loss_fn)

    hdf5_file = h5py.File('../u_du_Ca40_Sky3D_normed_sampled6sensors_full.h5', 'r')
    ndata = hdf5_file['Input'].shape[0]
    
    start = time.time()
    model_opt.fit(generator(hdf5_file, batch_size=batch_size), epochs=nepoch, steps_per_epoch=int(ndata/batch_size),verbose=1, callbacks=callbacks, use_multiprocessing=False)
    end = time.time()

    print("Training Time: ", end-start)
