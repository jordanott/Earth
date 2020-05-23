#main.py
import os
import pprint
import socket
import argparse
import subprocess
from stored_dictionaries.data_options import data_opts

parser = argparse.ArgumentParser()
parser.add_argument('--trial_id')
parser.add_argument('--output_dir')
FLAGS = parser.parse_args()

os.environ['SHERPA_TRIAL_ID']   = FLAGS.trial_id
os.environ['SHERPA_OUTPUT_DIR'] = FLAGS.output_dir

print('Trial', FLAGS.trial_id)
print('Host', socket.gethostname())
print(
    subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv'], stderr=subprocess.STDOUT)
)
print(
    subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv'], stderr=subprocess.STDOUT)
)
    
############################################################################
############################## SHERPA ######################################
import sherpa

client = sherpa.Client()
trial = client.get_trial()

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(trial.parameters)
############################################################################
############################################################################

############################################################################
############################ TENSORFLOW ####################################
import tensorflow as tf

# print(
#     subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
# )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
config.log_device_placement = False
session = tf.compat.v1.Session(config=config)
############################################################################
############################################################################

############################################################################
########################## DataGenerator ###################################
from data_generator import DataGenerator, Normalizer

norm = Normalizer(
    data_dir=trial.parameters['data_dir'] + trial.parameters['data'] + '/',
    norm_fn=data_opts[trial.parameters['data']]['norm_fn'],
    fsub='feature_means',
    fdiv='feature_stds',
    tmult='target_conv',
)

train_gen = DataGenerator(
    data_dir=trial.parameters['data_dir'] + trial.parameters['data'] + '/',
    feature_fn=data_opts[trial.parameters['data']]['train']['feature_fn'],
    target_fn=data_opts[trial.parameters['data']]['train']['target_fn'],
    batch_size=trial.parameters['batch_size'],
    transform=norm,
    shuffle=True,
)

valid_gen = DataGenerator(
    data_dir=trial.parameters['data_dir'] + trial.parameters['data'] + '/',
    feature_fn=data_opts[trial.parameters['data']]['test']['feature_fn'],
    target_fn=data_opts[trial.parameters['data']]['test']['target_fn'],
    batch_size=trial.parameters['batch_size'],
    transform=norm,
    shuffle=True,
    val=True
)

print('Train', train_gen.n_samples, train_gen.n_batches, flush=True)
print('Test', valid_gen.n_samples, valid_gen.n_batches, flush=True)

############################################################################
############################################################################
    
############################################################################
############################ Train Model ###################################
from model import Network

net = Network(trial.parameters, trial.id)

net.train(
    train_gen, 
    valid_gen, 
    trial=trial, 
    client=client
)
############################################################################
############################################################################

print ('Done with trial ' + str(trial.id) )