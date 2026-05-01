# PCL-FT Training Record — Flights_1 (E6/E7)

Standard LoRA fine-tuning on the 3-way PCL-labelled Flights_1 dataset.
Run 2026-04-24 on RunPod A100 80GB PCIe via
`scripts/runpod_pcl_ft.py --stage all`.

All configuration is the `src/train.py` defaults unless noted below.

## Shared configuration

| Item | Value |
|------|-------|
| Training data | `data/Flights_1_train_pcl.jsonl` (250 records; 3-way `refundable`) |
| Test data | `data/Flights_1_test_pcl.jsonl` (50 records) |
| Schema | `data/Flights_1_schema_pcl.json` (15 fields; `refundable` enum = `["True","False","ambiguous"]`) |
| Fine-tuning method | Standard LoRA (PEFT) — no custom adapter code |
| LoRA rank / alpha | 16 / 32 |
| LoRA targets | `q_proj,v_proj` |
| Epochs | 5 |
| Learning rate | 1e-4 (AdamW) |
| Batch size | 1 |
| Max seq length | 2048 |
| Precision | bfloat16 |
| Gradient checkpointing | on |
| Device | CUDA (A100 80GB) |

## Per-scale training

### 0.5B — `Qwen/Qwen2.5-0.5B-Instruct`

- Trainable LoRA params: **2.4M** (rank × layers adapter set)
- Training time: **223 s (3.7 min)**
- Checkpoint: `checkpoints/pcl_qwen0.5b/lora_epoch5/`

Loss curve (per-epoch mean over non-masked tokens):

| Epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| loss | (see log) | (see log) | (see log) | (see log) | (see log) |

(Full numbers in `runpod_pcl_0.5b.log` / `runpod_pcl_7b_32b.log` on RunPod and Mac.)

### 7B — `Qwen/Qwen2.5-7B-Instruct`

- Trainable LoRA params: **5,046,272** (0.066% of 7.62B total)
- Training time: **283 s (4.7 min)**
- Loss per epoch: 0.1808 → 0.1042 → 0.0993 → 0.0943 → 0.0888
- Checkpoint: `checkpoints/pcl_qwen7b/lora_epoch5/`

### 32B — `Qwen/Qwen2.5-32B-Instruct`

- Trainable LoRA params: **16,777,216** (0.051% of 32.78B total)
- Training time: **1078 s (18 min)**
- Loss per epoch: 0.1589 → 0.1014 → 0.0945 → 0.0888 → 0.0819
- Checkpoint: `checkpoints/pcl_qwen32b/lora_epoch5/`

**No OOM at 32B with gradient checkpointing** on an A100 80GB PCIe.

## Evaluation headlines

Forced-commit accuracy on 50-example `Flights_1_test_pcl.jsonl` across
5 constrained fields (250 field observations total):

| Scale | Baseline (IT-1 not applied) | PCL-FT (argmax) | Δ |
|---|---:|---:|---:|
| 0.5B | 43% | **89.6%** | **+47 pp** |
| 7B | 70.8% | **90.4%** | **+20 pp** |
| 32B | 71.6% | **92.4%** | **+21 pp** |

All three scales converge to ≈0.08 training loss and ≈90% test accuracy.
The +130% boolean regression the paper documents under 2-way labels at
32B is eliminated by 3-way PCL relabelling at every scale tested.

## Reproducing

```bash
# On a GPU host:
cd valid-json-wrong-answer
source venv/bin/activate
export HF_HOME=/workspace/hf_cache   # or wherever your HF cache lives
pip install peft

# Full sweep (0.5B, then 7B, then 32B)
python scripts/runpod_pcl_ft.py --stage all 2>&1 | tee runpod_pcl.log

# Just one scale
python scripts/runpod_pcl_ft.py --scales 7b --stage all
```

Outputs:

- `checkpoints/pcl_qwen<scale>/lora_epoch5/` — LoRA adapters
- `results/margin_gating/flights_qwen<scale>_pcl.json` — per-record confidence distributions for downstream margin-gating analysis

## Links

- E6 ledger entry: see `Experiment Plan.md`
- Per-record results for eval: `results/margin_gating/flights_qwen0.5b_pcl.json`, `flights_qwen7b_pcl.json`, `flights_qwen32b_pcl.json`
- Mac-side analysis recipe: see comments at top of `scripts/margin_gating_eval.py`
