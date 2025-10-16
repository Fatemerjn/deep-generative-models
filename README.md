# Deep Generative Models

Course assignments and utilities for the Deep Generative Models module. The repository now follows a
conventional Python project layout so experiments can be reproduced from scripts as well as notebooks.

## Project Layout

```
├── configs/                # YAML experiment definitions
├── docs/                   # Additional documentation
├── notebooks/
│   ├── practicals/         # In-class / exploratory notebooks
│   └── submissions/        # Submitted homework versions
├── scripts/                # Command-line entry points (training, evaluation)
├── src/deep_generative_models/
│   ├── data/               # Datamodule + dataset registry
│   ├── models/             # GAN, VAE, Flow implementations
│   ├── training/           # Training/evaluation loops
│   └── utils/              # Logging, plotting, checkpoint helpers
├── data/                   # Raw and processed datasets (gitignored)
└── results/                # Outputs, checkpoints, figures (gitignored)
```

## Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the project in editable mode (adds `src/` to `PYTHONPATH` automatically):

   ```bash
   pip install -e .
   ```

   If you prefer a minimal install for notebook work only:

   ```bash
   pip install -r requirements.txt
   export PYTHONPATH=$PWD/src  # Needed for importing the package without `pip install -e .`
   ```

## Usage

- **Notebooks**: Launch Jupyter Lab and open the desired file from `notebooks/practicals/` or
  `notebooks/submissions/`.

  ```bash
  jupyter lab
  ```

- **Training from the CLI**: Use the experiment scripts to run training/evaluation with a YAML config.

  ```bash
  # Train a VAE on MNIST (see configs/ for more examples)
  python scripts/train.py --config configs/example_mnist_vae.yaml

  # Evaluate a saved checkpoint
  python scripts/evaluate.py \
      --config configs/example_mnist_vae.yaml \
      --checkpoint results/mnist_vae/checkpoint_0010.pt
  ```

## Notes

- Keep large datasets and model checkpoints out of git; place them in `data/` and `results/`.
- Feel free to add new configs under `configs/` to track hyper-parameter sweeps.
- Helper modules under `src/deep_generative_models/` can be imported directly within notebooks, e.g.
  `from deep_generative_models.utils import plot_samples`.


