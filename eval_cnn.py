
# eval_cnn.py — Evaluate the final trained CNN on test images

import os
import json
import argparse
import torch

from cnn import load_all, norm, Net, predict_proba, get_device
from diagnostics import compute_metrics


def evaluate(model_path='models/face_net_final.pt',
             config_path='models/config_final.json',
             data_dir='./test_images',
             cache_file='./cache_test.pt',
             threshold=0.50):

    device = get_device()

    print("=" * 60)
    print("FACE DETECTION CNN — TEST EVALUATION")
    print("=" * 60)
    print(f"Device:     {device}")
    print(f"Model:      {model_path}")
    print(f"Threshold:  {threshold}")

    # Load configuration
    c1, c2 = 16, 32

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
            c1 = cfg.get('c1', c1)
            c2 = cfg.get('c2', c2)

    # Initialize model and load weights
    model = Net(c1=c1, c2=c2).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at '{model_path}'.")

    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=device)
    )

    model.eval()

    # Load test data
    print(f"Loading data from '{data_dir}'...")

    test_x, test_y = load_all(data_dir, cache_file)
    test_norm_x = norm(test_x.float() / 255.0)

    test_ds = torch.utils.data.TensorDataset(test_norm_x, test_y)

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False
    )

    # Predict probabilities
    y_true, y_prob = predict_proba(model, test_loader, device)

    metrics = compute_metrics(
        y_true,
        y_prob,
        threshold=threshold
    )

    conf = metrics['confusion']

    tp, fp, fn, tn = (
        conf['tp'],
        conf['fp'],
        conf['fn'],
        conf['tn']
    )

    total_faces = tp + fn
    total_nofaces = tn + fp
    total_images = len(y_true)

    raw_acc = (tp + tn) / total_images

    # Print results
    print("\n" + "-" * 60)
    print(f"RESULTS ON TEST SET ({total_images:,} images):")
    print("-" * 60)

    print(
        f"  • Face Recall (faces detected)                : "
        f"{metrics['recall_face'] * 100:.2f}% ({tp}/{total_faces})"
    )

    print(
        f"  • No-Face Recall (non-faces rejected)         : "
        f"{metrics['recall_noface'] * 100:.2f}% ({tn}/{total_nofaces})"
    )

    print(
        f"  • Face Precision                             : "
        f"{metrics['precision_face'] * 100:.2f}% ({tp}/{tp + fp})"
    )

    print(
        f"  • Balanced Accuracy (average of recalls)      : "
        f"{metrics['balanced_acc'] * 100:.2f}%"
    )

    print(
        f"  • Overall Accuracy                           : "
        f"{raw_acc * 100:.2f}% ({tp + tn}/{total_images})"
    )

    print(
        f"  • AUC-ROC                                    : "
        f"{metrics['auc']:.4f}"
    )

    print("-" * 60)

    print("CONFUSION MATRIX:")

    print(f"  True Positives  (Faces correctly detected)    : {tp}")
    print(f"  False Negatives (Faces not detected)          : {fn}")
    print(f"  True Negatives  (Non-faces correctly rejected): {tn}")
    print(f"  False Positives (Backgrounds classified as face): {fp}")

    print("=" * 60 + "\n")

    return metrics


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate final face classifier on test images"
    )

    parser.add_argument(
        '--model',
        default='models/face_net_final.pt',
        help='Path to model weights'
    )

    parser.add_argument(
        '--config',
        default='models/config_final.json',
        help='Path to model config'
    )

    parser.add_argument(
        '--data',
        default='./test_images',
        help='Path to test images folder'
    )

    parser.add_argument(
        '--cache',
        default='./cache_test.pt',
        help='Path to test cache file'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.50,
        help='Decision threshold (default: 0.50)'
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        config_path=args.config,
        data_dir=args.data,
        cache_file=args.cache,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()
