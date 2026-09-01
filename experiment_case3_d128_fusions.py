#!/usr/bin/env python3
"""Run the D128 fusion tournament at case 3's B=4 shape."""

import experiment_case2_d128_fusions as experiment
import torch_transformer_benchmark as bench


experiment.CONFIG = bench.TransformerConfig(4, 128, 128, 4, 128, 4, True)


if __name__ == "__main__":
    raise SystemExit(experiment.main())
