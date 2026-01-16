# Complete Weierstrass elliptic function solutions and canonical coordinates for four-wave mixing in nonlinear optical fibres

This repo contains the code and latex for the paper called _"Complete Weierstrass elliptic function solutions and canonical coordinates for four-wave mixing in nonlinear optical fibres"_

## Code

The code contains the following Python Jupyter Notebooks:

- `Four Wave Mixing Case.ipynb` - this is the main derivation of the Weierstrass elliptic function analytic solutions to four-wave mixing. It uses Python `SymPy` for symbolic maths.
- `The canonical coordinates of FWM.ipynb` - this is the main derivation of the canonical coordinates for four-wave mixing. It uses Python `SymPy` for symbolic maths and follows on from the previous notebook that derives solutions.
- `Numeric plots for four-wave mixing.ipynb` - this evaluates analytic solutions numerically to produce plots. It uses `mpmath`, `scipy`, and a slightly customised fork of `pyweierstrass`.

Requires Python 3.13. To create a virtual environment and install the required packages run the following:

```
cd code
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Latex

A VS code plugin was used to write the latex and the open source `Zotero` for references.
