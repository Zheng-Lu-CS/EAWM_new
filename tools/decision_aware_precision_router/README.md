# Decision-Aware Precision Router (DAPR)

This directory contains launch and diagnostic utilities for the fixed-slot
Decision-Aware Precision Router experiment.

DAPR keeps the image-token count unchanged. Each spatial slot receives either
the original high-precision token embedding or a low-precision local summary
embedding. This avoids changing the RetNet block shape while testing whether
decision-related signals can allocate visual precision better than a fixed grid.

Core implementation:

- `EASimulus/src/mechanisms/decision_aware_precision_router/`
- `EASimulus/config/mechanism/decision_aware_precision_router.yaml`

Default behavior is unchanged because `enabled=False`.

Useful overrides:

```bash
mechanism.decision_aware_precision_router.enabled=True
mechanism.decision_aware_precision_router.target_keep_ratio=0.5
mechanism.decision_aware_precision_router.router=gumbel
mechanism.decision_aware_precision_router.summary=local_mean
```

Recommended order:

0. Run the implementation smoke test:

```bash
python tools/decision_aware_precision_router/smoke_test_router.py
```

This does not require Atari/Craftax dependencies. It checks DAPR forward,
regularization, backward, and eval paths.

1. Run the offline diagnostic on an existing Atari baseline run:

```bash
RUN_DIR=/path/to/EASimulus/output/run \
DEVICE=cuda:0 \
bash tools/decision_aware_precision_router/run_patch_importance_diagnostic.sh
```

The diagnostic compares random, token-change/event, epistemic uncertainty,
reward-gradient, value-gradient, combined decision score, and oracle CE patch
selection under several keep ratios. If oracle or decision-score top-k does not
beat random, do not launch the full training sweep.

2. Run the Atari validation sweep:

```bash
ENV_NAME=your_conda_env \
PROJECT_ROOT=/path/to/EAWM \
bash tools/decision_aware_precision_router/run_atari_dapr_validation_4gpu.sh
```

Default tasks are `Breakout Seaquest Frostbite Kangaroo RoadRunner PrivateEye`
with seeds `0 1 2`. Override `TASKS`, `SEEDS`, `GPU_IDS`, `KEEP_RATIO`, or
`EXP_NAME` as needed.

3. Run Craftax sanity training:

```bash
ENV_NAME=your_conda_env \
PROJECT_ROOT=/path/to/EAWM \
bash tools/decision_aware_precision_router/run_craftax_dapr_4gpu.sh
```

Craftax is vector-observation based in this repository, so DAPR is expected to
be a no-op there. This script verifies that enabling the mechanism does not
break non-image benchmarks.
