"""
run_experiments.py — Automated runner for the simplified roadmap (seed 42, 1 change at a time).
"""

import os
import sys

# Ensure root directory is in sys.path when running from experiments/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cnn import CONFIG, run_experiment


def run_experiment_by_name(name):
    cfg = CONFIG.copy()
    cfg['seed'] = 42

    if name in ('step0', 'baseline'):
        # Step 0 baseline: SGD 0.001, momentum 0.9, 10 ep
        cfg['optimizer'] = 'sgd'
        cfg['lr'] = 0.001
        cfg['momentum'] = 0.9
        cfg['epochs'] = 10
        cfg['scheduler'] = 'none'
        cfg['min_scale'] = 0.9
        cfg['c1'], cfg['c2'] = 6, 16
        cfg['pooling'] = 'max'
        cfg['tag'] = 'step0'
        cfg['checkpoint_path'] = "./models/face_net_step0.pt"
        cfg['log_path'] = "./logs/training_log_step0.json"
        cfg['plot_path'] = "./figures/training_curves_step0.png"
        return run_experiment(cfg)

    elif name == 'a':
        # (a) 
        cfg['optimizer'] = 'nesterov'
        cfg['lr'] = 0.01
        cfg['epochs'] = 10
        cfg['scheduler'] = 'none'
        cfg['min_scale'] = 0.9
        cfg['c1'], cfg['c2'] = 6, 16
        cfg['pooling'] = 'max'
        cfg['tag'] = 'exp_a_nesterov'

    elif name == 'b':
        # (b)
        cfg['optimizer'] = 'adam'
        cfg['lr'] = 0.001
        cfg['epochs'] = 10
        cfg['scheduler'] = 'none'
        cfg['min_scale'] = 0.9
        cfg['c1'], cfg['c2'] = 6, 16
        cfg['pooling'] = 'max'
        cfg['tag'] = 'exp_b_adam'

    elif name == 'c':
        # (c) 
        cfg['optimizer'] = 'nesterov' 
        cfg['lr'] = 0.01
        cfg['epochs'] = 20
        cfg['scheduler'] = 'steplr'
        cfg['min_scale'] = 0.9
        cfg['c1'], cfg['c2'] = 6, 16
        cfg['pooling'] = 'max'
        cfg['tag'] = 'exp_c_steplr'

    elif name == 'd':
        # (d)
        cfg['optimizer'] = 'nesterov'
        cfg['lr'] = 0.01
        cfg['epochs'] = 20
        cfg['scheduler'] = 'steplr'
        cfg['min_scale'] = 0.7
        cfg['c1'], cfg['c2'] = 6, 16
        cfg['pooling'] = 'max'
        cfg['tag'] = 'exp_d_zoomout'

    elif name == 'e':
        # (e)
        cfg['optimizer'] = 'nesterov'
        cfg['lr'] = 0.01
        cfg['epochs'] = 20
        cfg['scheduler'] = 'steplr'
        cfg['min_scale'] = 0.7
        cfg['c1'], cfg['c2'] = 16, 32
        cfg['pooling'] = 'max'
        cfg['tag'] = 'exp_e_wider'

    elif name == 'f':
        # (f) 
        cfg['optimizer'] = 'nesterov'
        cfg['lr'] = 0.01
        cfg['epochs'] = 20
        cfg['scheduler'] = 'steplr'
        cfg['min_scale'] = 0.7
        cfg['c1'], cfg['c2'] = 16, 32
        cfg['pooling'] = 'avg'
        cfg['tag'] = 'exp_f_avgpool'

    else:
        print(f"Unknown experiment: {name}")
        return None

    cfg['checkpoint_path'] = f"./models/face_net_{cfg['tag']}.pt"
    cfg['log_path'] = f"./logs/training_log_{cfg['tag']}.json"
    cfg['plot_path'] = f"./figures/training_curves_{cfg['tag']}.png"

    return run_experiment(cfg)


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'b'
    run_experiment_by_name(name)
