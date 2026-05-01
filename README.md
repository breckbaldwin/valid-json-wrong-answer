# valid-json-wrong-answer

Per-grammar-role loss decomposition for evaluating structured JSON output.

This repo accompanies the paper *"Valid JSON, Wrong Answer: Fine-Tuning Degrades Schema Key Prediction at Scale"*.

## Key Findings

- Small baseline models degrade performance with grammar-contrained decoding. 
- Standard LoRA fine-tuning + grammar-constrained decoding produces valid JSON at all model scales. Aggregate loss metrics show clear improvement. But per-grammar-role decomposition reveals that **key prediction degrades** at 32B — the model memorizes training-set key ordering instead of learning the schema, while aggregate metrics hide the regression behind large gains on trivial structural tokens.

## Setup

```bash
git clone https://github.com/breckbaldwin/valid-json-wrong-answer.git
cd valid-json-wrong-answer
python -m venv venv #may be `python3 -m venv .venv`
source venv/bin/activate
pip install -r requirements.txt
```

### CUDA / PyTorch driver compatibility

`requirements.txt` pins PyTorch to the cu124 channel
(`--extra-index-url https://download.pytorch.org/whl/cu124`), which
works on any NVIDIA driver ≥ 12.4 — the default for current RunPod
A100 images. If your host has a different CUDA driver, override the
index URL when installing:

```bash
# Driver 12.1–12.3
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

# Driver 11.8–12.0
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

Symptom of a mismatch: `RuntimeError: The NVIDIA driver on your system
is too old (found version 1xxxx)` on the first `torch` import. Check
your driver with `nvidia-smi` and pick a CUDA channel ≤ that.

### HuggingFace token

**Only required for re-running experiments (Option B in the next
section). If you're verifying the paper from `results.tgz` (Option A),
you can skip this — no model downloads happen.**

You need a HuggingFace access token to download the Qwen 2.5 model
weights (the models are public but the Hub still requires
authentication for `from_pretrained` calls). Get one at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
— a Read token is sufficient.

Then either set it as an environment variable:

```bash
export HF_TOKEN=hf_...
```

or log in interactively (writes to `~/.cache/huggingface/token`):

```bash
huggingface-cli login
```

When provisioning a remote pod, set `HF_TOKEN` in the pod's environment
(e.g., add `export HF_TOKEN=hf_...` to `~/.bashrc`) before running any
script that downloads model weights.

## How to Reproduce

The paper's experiments are reproducible end-to-end via a single
orchestration script (`scripts/run_all_paper.sh`) on a fresh A100 80GB
RunPod instance. Total wall-clock budget is roughly 6 hours.

### Two paths

**Option A — verify the paper from saved results.** The repo ships
with `results.tgz` containing every experimental output referenced by
the paper. Unpack to inspect numbers without re-running anything:

```bash
tar xzf results.tgz
ls results/                       # per-grammar-role decomposition JSONs
ls results/margin_gating/         # margin-gating + PCL-FT outputs (.json + .md)
cat results/margin_gating/RESULTS.md
cat results/pcl_ft/TRAINING.md
```

`results.tgz` overlays into `results/` without disturbing the
re-run pipeline — every script in the orchestration is idempotent and
will skip outputs already on disk. You can mix and match: unpack the
tgz, then re-run individual phases to refresh specific cells.

**Option B — re-run from scratch on a fresh A100.** Start with a
RunPod A100 80GB PCIe (single GPU is sufficient for all three scales).
After cloning and `pip install -r requirements.txt`:

```bash
# Provision data (CUAD ships in the repo under data/cuad/CUAD_v1/;
# SGD is fetched and prepared by prepare_data.sh)
bash scripts/prepare_data.sh
python src/prepare_cuad.py

# Run all four phases. Idempotent — interrupt and resume safely.
bash scripts/run_all_paper.sh

# Or run one phase at a time
bash scripts/run_all_paper.sh phase1
bash scripts/run_all_paper.sh phase2 phase3 phase4
```

### Phases

| Phase | Wraps | Wall-clock | Produces (paper artefacts) |
|------:|------|-----------:|----------------------------|
| 1 | `scripts/run_experiment.sh all` | ~3.5 h | Per-grammar-role loss decomposition for Flights_1, Restaurants_1, CUAD × {0.5B, 7B, 32B} × {baseline, 10-epoch standard LoRA}. Backs `tab:aggregate`, `tab:flights_decomp`, `tab:restaurants_decomp`, `tab:enum_trend`. |
| 2 | `scripts/runpod_baseline.py --scale all` | ~30 min | Margin-gating baseline confidence distributions on all three datasets × three scales. Backs `tab:cuad_per_field_scale`, `tab:gating_flights`, `tab:margin_scaling`, and the baseline columns of `tab:head_to_head`. |
| 3 | `scripts/runpod_pcl_ft.py --datasets flights,restaurants,cuad` | ~90 min | PCL fine-tuning (3-way labels) + eval on all three datasets × three scales. Backs `tab:pcl` and the PCL-FT columns of `tab:head_to_head`. |
| 4 | `runpod_baseline.py` and `runpod_pcl_ft.py` with `--test-data-override data/cuad_test_full.jsonl --tag-suffix _full` | ~25 min | Full-context CUAD re-evaluation (truncation hypothesis test discussed in §6.2). |

### Outputs

```
results/
├── <scale>_{baseline,finetuned}_{flights,restaurants,cuad}.json   (per-role loss)
├── margin_gating/
│   ├── <dataset>_qwen<scale>{,_pcl}{,_full}.json                  (raw probabilities)
│   ├── <dataset>_qwen<scale>{,_pcl}{,_full}.md                    (per-tag analysis)
│   └── RESULTS.md                                                  (auto-generated index)
├── pcl_ft/TRAINING.md                                              (PCL-FT training record)
└── run_all/<phase>.log                                             (per-phase logs)

checkpoints/
├── <scale>_<dataset>_lora_epoch10/                                 (Phase 1 standard LoRA adapters)
└── <dataset>_pcl_qwen<scale>/lora_epoch5/                          (Phase 3 PCL-FT adapters)
```

### Local compute-side analysis after pod runs

After Phase 2 / 3 / 4 complete on a pod, rsync the `results/`
directory back, then regenerate the per-tag `.md` sections locally:

#### Table 1





```bash
# On Mac
rsync -avz user@pod:/path/valid-json-wrong-answer/results/ results/

# Generate per-tag analysis files (instant; no GPU needed)
for tag in $(ls results/margin_gating/*.json | xargs -n1 basename | sed 's/.json$//'); do
  case $tag in
    flights_*)     s=data/Flights_1_schema_pcl.json;     t=data/Flights_1_test_pcl.jsonl ;;
    restaurants_*) s=data/Restaurants_1_schema_pcl.json; t=data/Restaurants_1_test_pcl.jsonl ;;
    cuad_*_full)   s=data/cuad_schema.json;              t=data/cuad_test_full.jsonl ;;
    cuad_*)        s=data/cuad_schema.json;              t=data/cuad_test.jsonl ;;
  esac
  python scripts/margin_gating_eval.py --schema "$s" --data "$t" \
      --model _ --tag "$tag" --reuse-probs
done
```

`results/margin_gating/RESULTS.md` is regenerated automatically and
indexes every per-tag section.

### Reproducibility caveats

- LoRA training is **non-deterministic at bfloat16** unless seeds are
  pinned in `src/train.py` (and `transformers`/`torch` cuDNN flags are
  set). Numbers should land within ~1 percentage point of those in the
  paper; the qualitative findings (regression direction, margin shape,
  per-field rankings) are robust.
- HuggingFace model weights are content-addressed, so
  `Qwen/Qwen2.5-0.5B/7B/32B-Instruct` are bit-identical across pods —
  not a source of variance.
- Phase 4 requires the existing CUAD PCL-FT checkpoints from Phase 3.
  If you start fresh and skip Phase 3, Phase 4 will fail with a
  missing-checkpoint error.
- **CUAD train/test split size may differ by ±1 between snapshots.**
  `src/prepare_cuad.py` matches the CUAD `master_clauses.csv` rows
  against text files in the HF mirror by filename stem; the overlap
  has been observed at both 193 and 194 records on different machines,
  apparently due to filename-encoding quirks in `pathlib.rglob`. The
  adapter handles the drift automatically — `n_train` is preserved
  and `n_test` shrinks by 1 if needed (a `WARN:` line is printed).
  Difference is statistically immaterial (<0.5% of test set).

## Data Preparation

Uses two datasets:

- **Schema-Guided Dialogue (SGD)** — [GitHub](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue). Rastogi et al., AAAI 2020. License: CC BY-SA 4.0.
- **Contract Understanding Atticus Dataset (CUAD)** — [GitHub](https://github.com/TheAtticusProject/cuad). Hendrycks, Burns, Chen, Ball, 2021. License: CC BY 4.0. See [`data/cuad/ATTRIBUTION.md`](data/cuad/ATTRIBUTION.md) for full attribution and per-file licensing.

```bash
bash scripts/prepare_data.sh          # SGD (Flights_1, Restaurants_1)
python src/prepare_cuad.py            # CUAD — requires data/cuad/CUAD_v1/ staged first
```

## Local Usage

The Qwen2.5-0.5B-Instruct will probably run on a 8G laptop. 


```bash
# Per-grammar-role decomposition on a baseline model
python src/decompose.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --data data/Restaurants_1_test.jsonl \
    --schema data/Restaurants_1_schema.json \
    --device cpu

# Standard LoRA fine-tuning
python src/train.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --data data/Restaurants_1_train.jsonl \
    --epochs 5 --device cpu

# Decompose the fine-tuned model
python src/decompose.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --checkpoint checkpoints/lora_epoch5 \
    --data data/Restaurants_1_test.jsonl \
    --schema data/Restaurants_1_schema.json \
    --device cpu
```

## RunPod (GPU Experiments)

The paper's results require GPU for 7B and 32B models. We use [RunPod](https://www.runpod.io/) for GPU instances.

### Hardware Requirements

| Scale | GPU | VRAM | Approx Cost |
|-------|-----|------|-------------|
| 0.5B  | Any | 8GB+ | — |
| 7B    | A40 | 48GB | ~$0.39/hr |
| 32B   | A100 80GB | 80GB | ~$1.19/hr |

### Disk

When provisioning a RunPod instance, allocate **150 GB container disk
+ 150 GB volume disk** (the two-disk model RunPod presents at launch).
Breakdown:

- HuggingFace model cache for Qwen 2.5 {0.5B, 7B, 32B} weights: ~80 GB
  (32B alone is ~65 GB at bf16).
- Repo + data + intermediate training state: <5 GB.
- LoRA checkpoints accumulated across all phases: <2 GB.
- Headroom for `pip install`, transient files, logs: rest.

The container-disk allocation matters because the HF cache lives at
`/workspace/hf_cache` (set by the `HF_HOME` env var) on the volume
disk, but `pip install` and other ephemeral state goes to the
container disk; both need room.

### Launch and Setup

```bash
# Launch a pod (from your local machine)
python scripts/runpod_cloud.py launch --gpu "A100 PCIe" --name valid-json-wrong-answer

# SSH into the pod
python scripts/runpod_cloud.py ssh

# On the pod: run setup (provide your repo URL and HuggingFace token)
bash scripts/setup_runpod.sh <GIT_REPO_URL> <HF_TOKEN>
```

### Running Experiments

```bash
# Smoke test (check GPU, libraries, data, model cache)
bash scripts/smoketest.sh

# Run all scales (32B first to fail fast on memory issues)
bash scripts/run_experiment.sh all

# Or run one scale at a time
bash scripts/run_experiment.sh 32b
bash scripts/run_experiment.sh 7b
bash scripts/run_experiment.sh 05b

# Summarize results into comparison tables
python scripts/summarize_results.py
```

Each scale runs: baseline decomposition → LoRA training (10 epochs) → fine-tuned decomposition, on both Restaurants_1 and Flights_1 schemas.

## Repository Structure

```
src/
  prepare_data.py    — Extract SGD dialogue→JSON pairs, build JSON schemas
  train.py           — Standard LoRA fine-tuning (PEFT, no custom adapters)
  decode.py          — Constrained JSON generation (llguidance)
  decompose.py       — Per-grammar-role loss decomposition (post-hoc)
  evaluate.py        — Evaluation: exact match, ROUGE-L, key coverage
scripts/
  smoketest.sh       — RunPod environment verification
  run_experiment.sh  — Run all experiments (decomposable by scale)
  setup_runpod.sh    — One-shot pod setup
  summarize_results.py — Aggregate results into paper tables
  runpod_cloud.py    — RunPod pod management (launch, ssh, stop, etc.)
data/
  sgd/               — Schema-Guided Dialogue dataset
  *_train.jsonl      — Training data (250 examples per schema)
  *_test.jsonl       — Test data (50 examples per schema)
  *_schema.json      — JSON Schema for constrained decoding
```

## Grammar Roles

The decomposition assigns each token in the generated JSON to one of:

| Role | Description | Examples |
|------|-------------|----------|
| STRUCTURAL | JSON syntax | `{` `}` `[` `]` `:` `,` |
| QUOTE | String delimiters | `"` |
| KEY | Object key characters | `city`, `cuisine`, `price_range` |
| ENUM_VALUE | Categorical values | `moderate`, `Italian`, `Economy` |
| BOOLEAN | Boolean strings | `True`, `False` |
| NUMBER | Numeric characters | `364`, `2` |
| FREE_TEXT | Non-categorical content | restaurant names, addresses |
| WHITESPACE | Formatting | spaces, newlines |

## Citation

```bibtex
@article{baldwin2026validjson,
  title={Valid JSON, Wrong Answer: Fine-Tuning Degrades Schema Key Prediction at Scale},
  author={Baldwin, Breck},
  year={2026},
  note={arXiv preprint}
}
```

## License

Code: MIT. Data: SGD dataset is CC BY-SA 4.0 (Google Research).
