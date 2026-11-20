# Run artifacts for review tickets 09-13

Every number in RESULTS.md Part VIII comes from a log in this directory. They are
committed rather than left in `/tmp` so a published figure can be traced to the
run that produced it.

**All runs are pinned to 6 CPU threads** (`--threads 6`, the default in
`experiments/threads.py`). See RESULTS.md VIII.0 defect 2 for why: the DG /
SEPARATED key is not reproducible across thread counts. Each log records the
count actually in force on its first line.

| file | produced by | used in |
|---|---|---|
| `sweep_locomo_t6.{log,json}` | `separation_beta_sweep.py --corpus locomo --locomo 3` | VIII.1 |
| `sweep_qmsum_t6.{log,json}` | `separation_beta_sweep.py --corpus qmsum --qmsum 25` | VIII.1 |
| `sweep_locomo_dim256_t6.{log,json}` | same, `--dim 256 --betas 8 128` | VIII.6 (cause separation) |
| `completion_locomo_t6.{log,json}` | `completion_check.py --corpus locomo --locomo 10` | VIII.2 |
| `completion_qmsum_t6.{log,json}` | `completion_check.py --corpus qmsum --qmsum 25` | VIII.2 |
| `gist_locomo_t6.{log,json}` | `gist_check.py --corpus locomo --locomo 10` | VIII.3 |
| `gist_qmsum_t6.{log,json}` | `gist_check.py --corpus qmsum --qmsum 25` | VIII.3 |
| `abstention_recheck_t6.log` | `abstention_recheck.py --locomo 10 --betas 8 32 128 --protocol both` | VIII.4 |
| `kv_learned_readin_lr01_t6.{log,json}` | `kv_learned_readin.py --situations 16 --holdout 4 --steps 60 --lr 0.01` | VIII.5 |
| `kv_lrprobe_{0.03,0.01,0.003}_t6.log` | same at three learning rates, `--pools 1` | VIII.5 (hyperparameter robustness) |
| `kv_learned_readin_t6.log` | same at `--lr 0.1` | VIII.5 — **the diverged run**, kept as the record of why lr was re-selected |
| `recall_check_locomo_t6.log` | `recall_check.py --corpus locomo --locomo 3` | VIII.6 |
| `recall_check_qmsum_t6.log` | `recall_check.py --corpus qmsum --qmsum 6` | VIII.6 |
| `separation_check_{locomo,qmsum}_t6.log` | `separation_check.py --corpus <c>` | VIII.1 cross-check |
| `sweep_{locomo,qmsum}_t32.{log,json}` | the earlier **32-thread** ticket-09 run | VIII.0 defect 2 — the immunity comparison |
| `recall_check_locomo_t6_prior.log` | an earlier 6-thread run from the previous session | reproducibility cross-check |

The `_t32` files predate thread pinning and have no `threads` field in their JSON.
They are kept deliberately: comparing them against their `_t6` counterparts is the
measurement that demonstrates defect 2, and deleting them would remove the
evidence.
