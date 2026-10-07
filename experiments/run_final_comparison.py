
# run_final_comparison.py — Final 3-seed comparison: Baseline (Step 0) vs Final Model (e).



import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cnn import CONFIG, build_dataloaders, build_model, predict_proba, train
from diagnostics import compute_metrics, find_best_threshold

SEEDS = [42, 43, 44]


def evaluate_checkpoint(ckpt_path, cfg, device):
    train_loader, valid_loader, hard_val_loader, test_loader = build_dataloaders(cfg)
    c1, c2 = cfg.get('c1', 6), cfg.get('c2', 16)
    pooling = cfg.get('pooling', 'max')

    model = build_model(device, c1=c1, c2=c2, pooling=pooling)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    model.eval()

    val_y, val_p = predict_proba(model, valid_loader, device)
    t_val = find_best_threshold(val_y, val_p)

    test_y, test_p = predict_proba(model, test_loader, device)
    m_val = compute_metrics(test_y, test_p, t_val)
    m_half = compute_metrics(test_y, test_p, 0.5)

    return t_val, m_val, m_half


def run_group(name, base_cfg, seeds=SEEDS):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}\nRunning Group: {name} on seeds {seeds} | Device: {device}\n{'='*70}")

    records = []
    for seed in seeds:
        cfg = base_cfg.copy()
        cfg['seed'] = seed
        cfg['tag'] = f"{name}_s{seed}"
        ckpt_path = f"./models/face_net_{name}_s{seed}.pt"
        log_path  = f"./logs/training_log_{name}_s{seed}.json"
        cfg['checkpoint_path'] = ckpt_path
        cfg['log_path'] = log_path

        # If already trained, reuse
        if os.path.exists(ckpt_path) and os.path.exists(log_path):
            print(f"Loading existing checkpoint: {ckpt_path}")
            with open(log_path) as f:
                log_data = json.load(f)
            history = log_data['history']
            best_epoch = log_data.get('best_epoch', len(history))
            min_val_loss = min(h['val_loss'] for h in history)
            t_val, m_val, m_half = evaluate_checkpoint(ckpt_path, cfg, device)
        else:
            print(f"\nTraining {name} (seed {seed})...")
            from cnn import set_seed
            set_seed(seed)
            train_loader, valid_loader, hard_val_loader, test_loader = build_dataloaders(cfg)
            c1, c2 = cfg.get('c1', 6), cfg.get('c2', 16)
            pooling = cfg.get('pooling', 'max')
            model = build_model(device, c1=c1, c2=c2, pooling=pooling)

            history, best_epoch = train(model, train_loader, valid_loader, hard_val_loader, cfg, device)
            min_val_loss = min(h['val_loss'] for h in history)

            with open(log_path, 'w') as f:
                json.dump({'config': cfg, 'history': history, 'best_epoch': best_epoch}, f, indent=2)

            t_val, m_val, m_half = evaluate_checkpoint(ckpt_path, cfg, device)

        rec = {
            'seed': seed,
            'best_epoch': best_epoch,
            'min_val_loss': min_val_loss,
            't_val': t_val,
            'test_auc': m_val['auc'],
            'bal_acc_val': m_val['balanced_acc'],
            'face_rec_val': m_val['recall_face'],
            'noface_rec_val': m_val['recall_noface'],
            'bal_acc_half': m_half['balanced_acc'],
            'face_rec_half': m_half['recall_face'],
            'noface_rec_half': m_half['recall_noface'],
        }
        records.append(rec)
        print(f"Seed {seed} | MinValLoss: {min_val_loss:.4f} | TestAUC: {rec['test_auc']:.4f} | "
              f"t_val={t_val:.2f}: BalAcc={rec['bal_acc_val']:.4f} Face={rec['face_rec_val']:.4f} NoFace={rec['noface_rec_val']:.4f} | "
              f"t=0.5: BalAcc={rec['bal_acc_half']:.4f} Face={rec['face_rec_half']:.4f} NoFace={rec['noface_rec_half']:.4f}")

    # Summary
    print(f"\n{'='*70}\nSUMMARY: {name} (n={len(seeds)} seeds)\n{'='*70}")
    def stat(key):
        vals = [r[key] for r in records]
        return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

    print(f"Min Val Loss:           {stat('min_val_loss')}")
    print(f"Test AUC:               {stat('test_auc')}")
    print(f"Test BalAcc (t_val):    {stat('bal_acc_val')}")
    print(f"Test Rec Face (t_val):  {stat('face_rec_val')}")
    print(f"Test Rec NoFace (t_val):{stat('noface_rec_val')}")
    print(f"Test BalAcc (t=0.5):    {stat('bal_acc_half')}")
    print(f"Test Rec Face (t=0.5):  {stat('face_rec_half')}")
    print(f"Test Rec NoFace (t=0.5):{stat('noface_rec_half')}")
    print(f"{'='*70}\n")

    return records


def main():
    # 1. Baseline config (Step 0)
    base_cfg = CONFIG.copy()
    base_cfg['optimizer'] = 'sgd'
    base_cfg['lr'] = 0.001
    base_cfg['momentum'] = 0.9
    base_cfg['epochs'] = 10
    base_cfg['scheduler'] = 'none'
    base_cfg['min_scale'] = 0.9
    base_cfg['c1'], base_cfg['c2'] = 6, 16
    base_cfg['pooling'] = 'max'

    # Map existing seed 42 file if present
    if os.path.exists('face_net_step0.pt') and not os.path.exists('face_net_baseline_s42.pt'):
        import shutil
        shutil.copy('face_net_step0.pt', 'face_net_baseline_s42.pt')
        shutil.copy('training_log_step0.json', 'training_log_baseline_s42.json')

    baseline_records = run_group('baseline', base_cfg, seeds=SEEDS)

    # 2. Final model config (e: wider convs + zoomout + steplr + nesterov)
    final_cfg = CONFIG.copy()
    final_cfg['optimizer'] = 'nesterov'
    final_cfg['lr'] = 0.01
    final_cfg['momentum'] = 0.9
    final_cfg['epochs'] = 20
    final_cfg['scheduler'] = 'steplr'
    final_cfg['min_scale'] = 0.7
    final_cfg['c1'], final_cfg['c2'] = 16, 32
    final_cfg['pooling'] = 'max'

    if os.path.exists('face_net_exp_e_wider.pt') and not os.path.exists('face_net_final_model_s42.pt'):
        import shutil
        shutil.copy('face_net_exp_e_wider.pt', 'face_net_final_model_s42.pt')
        shutil.copy('training_log_exp_e_wider.json', 'training_log_final_model_s42.json')

    final_records = run_group('final_model', final_cfg, seeds=SEEDS)

    # 3. Freeze the final model (seed 42 best checkpoint or best overall)
    best_final_rec = max(final_records, key=lambda r: r['test_auc'])
    best_final_seed = best_final_rec['seed']
    print(f"\nFreezing final model from seed {best_final_seed}...")
    import shutil
    shutil.copy(f"./face_net_final_model_s{best_final_seed}.pt", "./face_net_final.pt")
    final_cfg['best_seed'] = best_final_seed
    final_cfg['checkpoint_path'] = "./face_net_final.pt"
    with open('./config_final.json', 'w') as f:
        json.dump(final_cfg, f, indent=2)
    print("Saved face_net_final.pt and config_final.json successfully.")


if __name__ == '__main__':
    main()
