import os
import json
import logging
import argparse
import subprocess
from filelock import FileLock

# distributor_pid = os.getpid()

parser = argparse.ArgumentParser()
parser.add_argument('--trial_id_min', type=int)
parser.add_argument('--trial_id_max', type=int)
parser.add_argument('--output_dir')
FLAGS = parser.parse_args()

pids = []

for trial_id in range(FLAGS.trial_id_min, FLAGS.trial_id_max+1):
    
    command = 'python3.6 main.py --trial_id {trial_id} --output_dir {output_dir}'.format(
        trial_id=trial_id,
        output_dir=FLAGS.output_dir
    )
    
    outfile = open(
        os.path.join(FLAGS.output_dir, 'jobs/trial_%05d.out' % trial_id),
        'w'
    )

    pids.append([
        subprocess.Popen(
            [command], 
            shell=True, 
            stdout=outfile,
            stderr=outfile
        ), 
        trial_id,
        outfile
    ])
            

completed_trials_file_name = os.path.join(FLAGS.output_dir, 'completed_trials.json')

while len(pids) != 0:
    for i in range(len(pids)-1,-1,-1):
        if pids[i][0].poll() is not None:
            _, trial_id, f = pids.pop(i)
            
            # CLOSE OUT FILE
            f.close()
            
            logging.disable(logging.CRITICAL)
            
            with FileLock(completed_trials_file_name + ".lock"):
                try:
                    with open(completed_trials_file_name, 'r') as f:
                        completed_trials = json.load(f)
                except:
                    completed_trials = []
                    
                completed_trials.append(trial_id)

                with open(completed_trials_file_name, 'w') as f:
                    f.write(json.dumps(completed_trials))
                    
            logging.disable(logging.NOTSET)
            
            print('Triald', trial_id, 'completed', flush=True)
            
            
            
# with FileLock("output/distributors.json.lock"):
#     with open('output/distributors.json', 'w+') as f:
#         distributors = json.load(f)
#         distributors[distributor_pid] = trial_pids
        
#         f.seek(0) ; f.truncate()
#         f.write(distributors)
    
    