"""Analyze the left flank of the baseline PRC: purity of top-ranked pixels."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import main

device = torch.device("cuda:0")
data = main.load_lgg_dataset_patient_aware("lgg-mri-segmentation/kaggle_3m")
train, val, test = main.patient_holdout_split(data, seed=2026)[0]
_, _, loader = main.create_dataloaders(train, val, test, batch_size=16)

ckpt = torch.load("../checkpoints/500k_baseline/lgg/patient-holdout/single/swin/seed_2026/recipe_0_matched_500k_100ep_tversky_1a554a10.pt",
                  map_location="cpu", weights_only=False)
model = main.create_swin_qat("matched_500k", recipe_id=0).to(device)
model.load_state_dict(ckpt["state_dict"], strict=True)
res = main.run_validation_inference(model, loader, device)

p = np.asarray(res["all_probs"], dtype=np.float32).ravel()
t = np.asarray(res["all_targets"]).ravel().astype(bool)
print(f"pixels: {p.size}, positive rate: {t.mean():.4f}")

# purity among top-scored pixels
order = np.argsort(-p)
for k in (1_000, 10_000, 100_000, 1_000_000):
    top = t[order[:k]]
    print(f"top {k:>9,} pixels: precision = {top.mean():.4f}")

# the p>0.99 spike: how pure is it?
hi = p > 0.99
print(f"\np>0.99: {hi.sum():,} pixels ({hi.mean()*100:.2f}%), purity = {t[hi].mean():.4f}")
print(f"tumor pixels with p>0.99: {(t & hi).sum():,} / {t.sum():,} = {(t & hi).sum()/t.sum():.3f} of all tumor")

# per-slice breakdown: patient-level split -> unseen patients are hard
n_slice = 256 * 256
aps = []
from sklearn.metrics import average_precision_score
for i in range(0, p.size, n_slice):
    ps, ts = p[i:i+n_slice], t[i:i+n_slice]
    if ts.sum() > 0:
        aps.append(average_precision_score(ts, ps))
aps = np.array(aps)
print(f"\npositive test slices: {len(aps)}")
print(f"per-slice AP: mean {aps.mean():.3f}, median {np.median(aps):.3f}")
print(f"  AP > 0.9 : {(aps > 0.9).sum()} slices")
print(f"  AP < 0.3 : {(aps < 0.3).sum()} slices")
print(f"  AP < 0.1 : {(aps < 0.1).sum()} slices")
