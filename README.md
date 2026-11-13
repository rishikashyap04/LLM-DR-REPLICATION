# LLM-DR replication: persona-grounded LLM delivery riders

A replication of **"LLM-DR: An On-Demand Delivery Simulation Framework for Studying Heterogeneous Rider Behavior and Collective Mobility with Persona-Grounded Large Language Model Agents"** (Zhang et al., arXiv 2603.12559), built on the authors' released Beijing data and reference results: https://github.com/Nicholas0027/LLM-DR

The notebook rebuilds four rider personas from real data, simulates a dispatch platform on one day of real orders, lets an LLM play each rider (reject, accept one order, or bundle several), and compares the outcome with the real data and with the authors' own runs.

## Files

| File | Purpose |
|---|---|
| `LLM_DR_replication.ipynb` | The notebook. Run the cells top to bottom. It writes `llm_dr.py` itself, so it is self-contained. |
| `llm_dr.py` | The simulator, personas, baselines, LLM policy and validation code (the same file the notebook writes). |

## Requirements

- Kaggle notebook with **GPU T4 x2** and **Internet ON** (needed for `pip`, `git clone` and downloading the model). A local machine needs a GPU able to serve a 7B 4-bit model with vLLM.
- About **1.5 hours** in total; the LLM run alone is about 50 minutes.
- No API key is needed. The model runs locally through vLLM.

## How to run

1. Open `LLM_DR_replication.ipynb` on Kaggle, set the accelerator to GPU T4 x2 and Internet to ON.
2. Run the cells in order, from Section 0 to Section 6. Keep the tab open during the LLM run.
3. After Section 4 finishes, run Section 5 for the results and Section 6 to zip them.
4. Download `results_backup.zip` before the session ends. Files in `/kaggle/working` are deleted when the session resets.

| Section | What it does |
|---|---|
| 0. Setup | Installs packages, sets paths, checks vLLM and the GPU. |
| 1. Data and code | Clones the authors' repo, writes `llm_dr.py`, builds `orders_clean.csv` (8,658 orders between 08:00 and 20:00 from 705 couriers). |
| 2. Routing | Calibrates the routing stand-in from the authors' reference waves (see Deviations). |
| 3. Baselines | Runs four rule-based riders: nearest order, highest fee, fee per distance, persona heuristic. Each run also rebuilds the personas and prints a check against the authors' assignments. |
| 4. LLM riders | Starts a local vLLM server with Qwen2.5-7B-Instruct (4-bit AWQ) and runs the persona-grounded simulation. |
| 5. Results | LLM against the authors' DeepSeek run, validation against real data, decision log, baselines against the authors' runs. |
| 6. Save | Zips the `results` folder. |

## Method in brief

- **Personas:** k-means (k=4) on the authors' 15 released rider features, giving Full-time Workhorse, Lunch-Peak Specialist, Super Stacker and Dinner-Peak Core.
- **Platform:** 10-minute decision steps, a 30-minute pending look-back, 2 km broadcast radius, each order offered to the 3 nearest idle riders, at most 6 offers per rider, 120 s service time per order. Conflicts go to the rider closest to the pickup.
- **Fleet:** 60 riders (15 per persona) starting at pickup locations sampled from the real orders.
- **LLM rider:** a prompt with persona description and statistics, current state, memory of the last 5 orders and the candidate orders. The model replies in JSON. Invalid order IDs are dropped, and an unparseable reply is retried once and then treated as a rejection.
- **Fee schedule:** CNY 3 / 6 / 10 / 15 for straight-line distances up to 1 km, 1-3 km, 3-5 km and over 5 km.
- **Evaluation:** per-persona delivery performance (Table I analogue) and distribution distances against the real data (JSD, Wasserstein-1, KS).

## Results of my run

Qwen2.5-7B-Instruct, one run, one day. Re-running gives slightly different LLM numbers (temperature 0.2, parallel threads).

- 1,536 of 8,658 orders completed (17.7%; paper 16.5%).
- Personas: silhouette 0.109 (paper 0.108), 99.4% agreement with the authors' assignments, group sizes 283 / 217 / 112 / 98 (paper 281 / 219 / 112 / 98).

| Persona | Orders per rider | Bundle size | Stacked ratio | Authors' DeepSeek run: orders / bundle / stacked |
|---|---|---|---|---|
| Full-time Workhorse | 24.9 | 1.512 | 0.464 | 26.3 / 1.521 / 0.521 |
| Lunch-Peak Specialist | 28.7 | 1.475 | 0.428 | 16.0 / 1.319 / 0.313 |
| Super Stacker | 21.9 | 1.637 | 0.512 | 30.4 / 2.235 / 0.828 |
| Dinner-Peak Core | 26.8 | 1.480 | 0.424 | 22.5 / 1.123 / 0.123 |

**Summary:** the data, persona and simulator parts replicate closely, and the rule-based baselines land close to the paper's. The LLM part replicates partly. The direction is right (Super Stacker bundles most and makes the fewest decisions), but the spread between personas is much narrower than in the paper, and Lunch-Peak accepts almost every offer. A likely cause is that a 7B model follows persona instructions more loosely than DeepSeek; this was not tested directly. The full tables, including validation and the decision log, are in the last cell of the notebook.

## Deviations from the paper

- **Model:** Qwen2.5-7B-Instruct (AWQ 4-bit, served with vLLM) instead of DeepSeek-v4-pro, because of budget. The simulator talks to any OpenAI-compatible endpoint (`--base_url`, `--model`), so the model can be swapped.
- **Routing:** the Amap cycling API was not used. Route length is the straight-line distance times a road factor (about 1.8, measured from the authors' Amap-routed reference waves) at 15 km/h. The notebook removes any `AMAP_KEY` from the environment so that every run uses this stand-in, and the Amap request counter stays at 0. Real Amap routing was not tested in this notebook.
- **Baselines:** the paper describes the four rules only briefly, so the exact thresholds are my own choices.

## Limitations

- One run on one day, without repeated seeds or significance tests; with 15 riders per persona, differences should be read as patterns and not as proof.
- The persona statistics and the evaluation data come from the same day (the paper notes this too), and order-time JSD is nearly the same for the rule-based baselines, so it does not by itself show that the personas were applied.
- Not reproduced: pickup-grid JSD, radius-of-gyration and area errors, the paper's figures, and the fleet-level temporal and spatial analysis.

## Troubleshooting

- **`pip` prints dependency-conflict warnings:** harmless for this notebook. Check the Section 0 output instead: it should show a vLLM version and `gpu ok: True`.
- **The server never becomes ready (Section 4):** run `!tail -40 /kaggle/working/vllm.log` and read the error. Some vLLM versions have trouble on T4 GPUs; pinning an older vLLM release can help.
- **`FileNotFoundError` for a results or data file:** the session was reset, so rerun from Section 0.
- **No Internet:** `pip`, `git clone` and the model download all need it.

## Credit

The data, personas, reference results and method are from the authors of the paper (Zhang, Jin, Zhang, Yang, Cui, Xiao and Yu) and their repository. This notebook is a reimplementation of the pipeline for study purposes.

