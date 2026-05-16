# SF-MI Bose-Hubbard Explorer

Streamlit webapp for exploring the superfluid--Mott insulator transition in the Bose-Hubbard model.

The app lets users choose:

- atomic species
- optical lattice wavelength
- lattice constant / spacing
- lattice depth
- lattice geometry
- central chemical potential
- harmonic trap frequency
- lattice size and local occupation cutoff

It outputs:

- analytic mean-field Mott lobes
- numerical uniform Gutzwiller phase diagrams
- site-dependent Gutzwiller wedding-cake density profiles
- order parameter maps
- first-order coherence visualizations
- simulated momentum/time-of-flight images
- small exact-diagonalization benchmarks and finite-size gap curves

## Run locally

```bash
conda create -n sfmi python=3.11
conda activate sfmi
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to:

```text
app.py
```

## Notes

The phase diagram uses uniform Gutzwiller mean-field theory. The real-space density, order parameter, and coherence maps use site-dependent Gutzwiller mean-field theory. The exact diagonalization tab is intentionally limited to small systems because the Hilbert space grows exponentially with system size.
