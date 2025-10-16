# Deep Generative Models

Course assignments and practical notebooks for Deep Generative Models.

Repository structure

- notebooks/        - Jupyter notebooks (home for all practical work)
- src/              - Python modules, training code and utilities
- data/             - Raw and processed datasets (gitignored)
- results/          - Generated outputs, figures and checkpoints (gitignored)
- docs/             - Additional documentation

Getting started

1. Create a virtual environment (recommended):

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Launch Jupyter Lab/Notebook:

   jupyter lab

Notes

- Keep large datasets and model checkpoints out of git; place them in `data/` and `results/` respectively and add to `.gitignore`.
- Each homework notebook has been moved to `notebooks/`.


