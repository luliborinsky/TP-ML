
#run_3seed_suite.py — 3-Seed Comparison Suite: Baseline vs (c) vs (e).



import os
import sys
import json
import numpy as np
import torch
import shutil

# Ensure root directory is in sys.path when running from experiments/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cnn import CONFIG, build_dataloaders, build_model, predict_proba, train, set_seed
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
    print(f"\n{'='*70}\nRunning Group: {name} (seeds {seeds}) | Device: {device}\n{'='*70}")

    records = []
    for seed in seeds:
        cfg = base_cfg.copy()
        cfg['seed'] = seed
        cfg['tag'] = f"{name}_s{seed}"
        ckpt_path = f"./models/face_net_{name}_s{seed}.pt"
        log_path  = f"./logs/training_log_{name}_s{seed}.json"
        cfg['checkpoint_path'] = ckpt_path
        cfg['log_path'] = log_path

        # If already trained with val_loss rule and exists, reuse
        if os.path.exists(ckpt_path) and os.path.exists(log_path):
            print(f"Loading existing checkpoint for {name} (seed {seed}): {ckpt_path}")
            with open(log_path) as f:
                log_data = json.load(f)
            history = log_data['history']
            best_epoch = log_data.get('best_epoch', len(history))
            min_val_loss = min(h['val_loss'] for h in history)
            t_val, m_val, m_half = evaluate_checkpoint(ckpt_path, cfg, device)
        else:
            print(f"\nTraining {name} (seed {seed})...")
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
        print(f"[{name}] Seed {seed} | MinValLoss: {min_val_loss:.4f} | TestAUC: {rec['test_auc']:.4f} | "
              f"t_val={t_val:.2f}: BalAcc={rec['bal_acc_val']:.4f} Face={rec['face_rec_val']:.4f} NoFace={rec['noface_rec_val']:.4f} | "
              f"t=0.5: BalAcc={rec['bal_acc_half']:.4f} Face={rec['face_rec_half']:.4f} NoFace={rec['noface_rec_half']:.4f}")

    def stat(key):
        vals = [r[key] for r in records]
        return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

    print(f"\n{'='*70}\nSUMMARY: {name} (n={len(seeds)} seeds)\n{'='*70}")
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
    # 1. Baseline (Step 0) config
    base_cfg = CONFIG.copy()
    base_cfg['optimizer'] = 'sgd'
    base_cfg['lr'] = 0.001
    base_cfg['momentum'] = 0.9
    base_cfg['epochs'] = 10
    base_cfg['scheduler'] = 'none'
    base_cfg['min_scale'] = 0.9
    base_cfg['c1'], base_cfg['c2'] = 6, 16
    base_cfg['pooling'] = 'max'

    # Re-use seed 42 of baseline if available from step0
    if os.path.exists('./models/face_net_step0.pt') and os.path.exists('./logs/training_log_step0.json'):
        shutil.copy('./models/face_net_step0.pt', './models/face_net_baseline_s42.pt')
        shutil.copy('./logs/training_log_step0.json', './logs/training_log_baseline_s42.json')

    # 2. Config (c) — StepLR
    c_cfg = CONFIG.copy()
    c_cfg['optimizer'] = 'nesterov'
    c_cfg['lr'] = 0.01
    c_cfg['momentum'] = 0.9
    c_cfg['epochs'] = 20
    c_cfg['scheduler'] = 'steplr'
    c_cfg['min_scale'] = 0.9
    c_cfg['c1'], c_cfg['c2'] = 6, 16
    c_cfg['pooling'] = 'max'

    # Re-use seed 42 of c if available
    if os.path.exists('./models/face_net_exp_c_steplr.pt') and not os.path.exists('./models/face_net_c_steplr_s42.pt'):
        shutil.copy('./models/face_net_exp_c_steplr.pt', './models/face_net_c_steplr_s42.pt')
        shutil.copy('./logs/training_log_exp_c_steplr.json', './logs/training_log_c_steplr_s42.json')

    # 3. Config (e) — Wider Convs + zoomout + steplr
    e_cfg = CONFIG.copy()
    e_cfg['optimizer'] = 'nesterov'
    e_cfg['lr'] = 0.01
    e_cfg['momentum'] = 0.9
    e_cfg['epochs'] = 20
    e_cfg['scheduler'] = 'steplr'
    e_cfg['min_scale'] = 0.7
    e_cfg['c1'], e_cfg['c2'] = 16, 32
    e_cfg['pooling'] = 'max'

    # Re-use seed 42 of e if available
    if os.path.exists('./models/face_net_exp_e_wider.pt') and not os.path.exists('./models/face_net_e_wider_s42.pt'):
        shutil.copy('./models/face_net_exp_e_wider.pt', './models/face_net_e_wider_s42.pt')
        shutil.copy('./logs/training_log_exp_e_wider.json', './logs/training_log_e_wider_s42.json')

    # Run groups
    baseline_records = run_group('baseline', base_cfg, seeds=SEEDS)
    c_records        = run_group('c_steplr', c_cfg, seeds=SEEDS)
    e_records        = run_group('e_wider', e_cfg, seeds=SEEDS)

    # Decision on final model strictly by mean min validation loss
    mean_val_loss_c = np.mean([r['min_val_loss'] for r in c_records])
    mean_val_loss_e = np.mean([r['min_val_loss'] for r in e_records])

    print(f"\nDecision by validation loss: (c) Mean MinValLoss = {mean_val_loss_c:.4f} vs (e) Mean MinValLoss = {mean_val_loss_e:.4f}")
    if mean_val_loss_c <= mean_val_loss_e:
        winner_name = 'c_steplr'
        winner_cfg  = c_cfg
        winner_recs = c_records
    else:
        winner_name = 'e_wider'
        winner_cfg  = e_cfg
        winner_recs = e_records

    print(f"Selected final model: {winner_name}")

    # Freeze final weights (strict min val loss)
    best_rec = min(winner_recs, key=lambda r: r['min_val_loss'])
    best_seed = best_rec['seed']
    print(f"Freezing weights from seed {best_seed} (val loss {best_rec['min_val_loss']:.4f})...")
    shutil.copy(f"./models/face_net_{winner_name}_s{best_seed}.pt", "./models/face_net_final.pt")
    winner_cfg['best_seed'] = best_seed
    winner_cfg['checkpoint_path'] = "./models/face_net_final.pt"
    with open('./models/config_final.json', 'w') as f:
        json.dump(winner_cfg, f, indent=2)
    print("face_net_final.pt and config_final.json saved successfully.")

    def format_row(label, recs):
        def m_s(k, pct=False):
            vals = [r[k] for r in recs]
            if pct:
                return f"{np.mean(vals)*100:.2f} ± {np.std(vals)*100:.2f}%"
            return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
        return (f"| {label} | {m_s('min_val_loss')} | {m_s('test_auc')} | "
                f"{m_s('bal_acc_val', True)} | {m_s('face_rec_val', True)} | {m_s('noface_rec_val', True)} | "
                f"{m_s('bal_acc_half', True)} | {m_s('face_rec_half', True)} | {m_s('noface_rec_half', True)} |")

    print("\n" + "="*80)
    print("FINAL 3-SEED COMPARISON SUMMARY TABLE (mean ± std, n=3 seeds: 42, 43, 44)")
    print("="*80)
    print("| Configuration | Min Val Loss | Test AUC | BalAcc (t_val) | Rec Face (t_val) | Rec NoFace (t_val) | BalAcc (t=0.5) | Rec Face (t=0.5) | Rec NoFace (t=0.5) |")
    print("|---------------|--------------|----------|----------------|------------------|--------------------|----------------|------------------|--------------------|")
    print(format_row("Baseline (Step 0)", baseline_records))
    print(format_row("(c) StepLR", c_records))
    print(format_row("(e) Wider Convs", e_records))
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
