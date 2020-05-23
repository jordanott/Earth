# runner.py
import os
import sherpa
import argparse
import itertools


parser = argparse.ArgumentParser()
parser.add_argument('--max_dense_layers', type=int, default=20, help='Max number of layers')
parser.add_argument('--data', type=str, default='32col', choices=['fluxbypass_aqua', 'land_data', '8col', '32col'])

parser.add_argument('--num_distributors', type=int, default=5, help='Number of gpus to occupy')
parser.add_argument('--num_tasks_per_distributor', type=int, default=1, help='Number of tasks per gpu')
FLAGS = parser.parse_args()


output_dir = 'SherpaResults/{data}/'.format(
    data=FLAGS.data
)

os.makedirs(
    os.path.join(output_dir, 'Models'), 
    exist_ok=True
)

parameters = [
    # LEARNING RATE
    sherpa.Continuous('lr', [0.00001, 0.1]),
    sherpa.Continuous('lr_decay', [0.5, 1.0]),
    
    # REGULARIZATION
    sherpa.Continuous('leaky_relu', [0., 0.9]),
    sherpa.Continuous('dropout', [0., 0.9]),
    sherpa.Continuous('noise', [0., 1.]),
    sherpa.Ordinal('batch_norm', [1, 0]),
    
    # ARCHITECTURE
    sherpa.Discrete('num_layers', [3,FLAGS.max_dense_layers]),
    sherpa.Discrete('num_dense_nodes', [128, 1024]),
    sherpa.Choice('optimizer', ['adam', 'sgd', 'rmsprop']),

    
    # SETTINGS
    sherpa.Choice('epochs', [30]),
    sherpa.Choice('batch_size', [16384]),
    sherpa.Choice('patience', [10]),
    sherpa.Choice('data', [FLAGS.data]),
    sherpa.Choice('data_dir', ['/oasis/projects/nsf/sio134/jott1/']),
    sherpa.Choice('model_dir', [output_dir + 'Models/']),
]

# sched = sherpa.schedulers.LocalScheduler(resources=resources)
sched = sherpa.schedulers.SLURMScheduler(
    submit_options='run.csh',
    environment='/home/jott1/.bashrc',
    username='jott1'
)
alg = sherpa.algorithms.RandomSearch(max_num_trials=10000)


sherpa.optimize(
    parameters=parameters,
    algorithm=alg,
    lower_is_better=True,
    command='python main.py',
    scheduler=sched,
    num_distributors=FLAGS.num_distributors,
    num_tasks_per_distributor=FLAGS.num_tasks_per_distributor,
    output_dir=output_dir
)