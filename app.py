"""Streamlit webapp for exploring the Bose-Hubbard SF-MI transition.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import streamlit as st
from scipy.spatial import distance_matrix

# =============================================================================
# Physical constants and lookup tables
# =============================================================================

h = 6.62607015e-34
amu = 1.66053906660e-27
a0 = 5.29177210903e-11

SPECIES_DATA = {
    "Rb87": {"mass_amu": 86.9091805, "a_s_a0": 100.4},
    "Cs133": {"mass_amu": 132.90545196, "a_s_a0": 280.0},
    "Na23": {"mass_amu": 22.98976928, "a_s_a0": 54.5},
    "K40": {"mass_amu": 39.96399848, "a_s_a0": 174.0},
}

GEOMETRIES = {
    "1D Chain": 2,
    "Honeycomb": 3,
    "Square": 4,
    "Triangular": 6,
}

# =============================================================================
# Experimental parameter mapping
# =============================================================================

@st.cache_data(show_spinner=False)
def calculate_hubbard_params(species: str, wavelength_nm: float, lattice_depth_er: float):
    """Estimate J, U, and E_R for a sinusoidal optical lattice.

    Uses common deep-lattice approximations. The returned J/U is dimensionless.
    """
    data = SPECIES_DATA[species]
    mass_kg = data["mass_amu"] * amu
    scattering_length_m = data["a_s_a0"] * a0
    wavelength_m = wavelength_nm * 1e-9
    s = lattice_depth_er

    recoil_energy_j = h**2 / (2 * mass_kg * wavelength_m**2)
    tunneling_j = (4 / np.sqrt(np.pi)) * s ** (3 / 4) * np.exp(-2 * np.sqrt(s)) * recoil_energy_j
    onsite_u_j = np.sqrt(8 / np.pi) * (2 * np.pi / wavelength_m) * scattering_length_m * s ** (3 / 4) * recoil_energy_j

    return {
        "J_over_U": float(tunneling_j / onsite_u_j),
        "U_over_J": float(onsite_u_j / tunneling_j),
        "J_J": float(tunneling_j),
        "U_J": float(onsite_u_j),
        "E_R_J": float(recoil_energy_j),
        "mass_kg": float(mass_kg),
        "wavelength_m": float(wavelength_m),
        "lattice_spacing_m": float(wavelength_m / 2),
    }


def print_hz(value_j: float) -> float:
    return value_j / h

# =============================================================================
# Mean-field theory and phase diagram
# =============================================================================

@st.cache_data(show_spinner=False)
def analytic_mott_lobe(mu_over_u: np.ndarray, n: int, z: int):
    """Analytic uniform mean-field Mott lobe boundary in J/U."""
    return (n - mu_over_u) * (mu_over_u - n + 1) / (z * (mu_over_u + 1))


def boson_operators(nmax: int):
    dim = nmax + 1
    b = np.diag(np.sqrt(np.arange(1, dim)), k=1)
    bdag = b.T
    n_op = bdag @ b
    return b, bdag, n_op

@st.cache_data(show_spinner=False)
def solve_single_site_mft(mu_over_u: float, j_over_u: float, z: int, nmax: int = 6):
    """Uniform Gutzwiller mean-field solver for one point in parameter space."""
    b, bdag, n_op = boson_operators(nmax)
    ident = np.eye(nmax + 1)
    n_int = n_op @ (n_op - ident)

    psi = 0.05 + 0j
    density = 0.0

    for _ in range(120):
        h_mf = (
            0.5 * n_int
            - mu_over_u * n_op
            - z * j_over_u * (np.conj(psi) * b + psi * bdag)
            + z * j_over_u * abs(psi) ** 2 * ident
        )
        eigvals, eigvecs = la.eigh(h_mf)
        ground_state = eigvecs[:, 0]
        new_psi = ground_state.conj().T @ b @ ground_state
        density = float(np.real(ground_state.conj().T @ n_op @ ground_state))
        if abs(new_psi - psi) < 1e-7:
            psi = new_psi
            break
        psi = 0.5 * new_psi + 0.5 * psi

    return float(abs(psi)), float(density)

@st.cache_data(show_spinner=False)
def calc_uniform_mft_grids(z: int, max_mu: float, nmax: int = 6):
    mu_vals = np.linspace(-0.1, max_mu, 65)
    j_vals = np.linspace(0.0005, 0.15, 55)
    psi_grid = np.zeros((len(mu_vals), len(j_vals)))
    density_grid = np.zeros_like(psi_grid)

    for i, mu in enumerate(mu_vals):
        for j, j_over_u in enumerate(j_vals):
            psi, den = solve_single_site_mft(mu, j_over_u, z, nmax=nmax)
            psi_grid[i, j] = psi
            density_grid[i, j] = den
    return j_vals, mu_vals, psi_grid, density_grid


def classify_point(j_over_u: float, mu_over_u: float, z: int, max_n: int = 5):
    for n in range(1, max_n + 1):
        if (n - 1) < mu_over_u < n:
            boundary = analytic_mott_lobe(np.array([mu_over_u]), n, z)[0]
            if j_over_u < boundary:
                return f"Mott insulator, n={n}"
    return "Superfluid"

# =============================================================================
# Geometry generation and site-dependent Gutzwiller
# =============================================================================

@st.cache_data(show_spinner=False)
def generate_lattice(geometry: str, lx: int, ly: int, lattice_spacing_m: float):
    """Return coordinates and nearest-neighbor list for supported lattices."""
    a = lattice_spacing_m
    coords: list[list[float]] = []

    if geometry == "1D Chain":
        coords = [[(ix - (lx - 1) / 2) * a, 0.0] for ix in range(lx)]
        neighbor_cutoff = 1.1 * a
    elif geometry == "Square":
        for iy in range(ly):
            for ix in range(lx):
                coords.append([(ix - (lx - 1) / 2) * a, (iy - (ly - 1) / 2) * a])
        neighbor_cutoff = 1.1 * a
    elif geometry == "Triangular":
        for iy in range(ly):
            for ix in range(lx):
                x = (ix + 0.5 * iy) * a
                y = (np.sqrt(3) / 2) * iy * a
                coords.append([x, y])
        coords = (np.array(coords) - np.mean(np.array(coords), axis=0)).tolist()
        neighbor_cutoff = 1.1 * a
    elif geometry == "Honeycomb":
        a1 = np.array([np.sqrt(3) * a, 0.0])
        a2 = np.array([np.sqrt(3) / 2 * a, 1.5 * a])
        delta = np.array([0.0, a])
        for iy in range(ly):
            for ix in range(lx):
                r = ix * a1 + iy * a2
                coords.append(r.tolist())
                coords.append((r + delta).tolist())
        coords = (np.array(coords) - np.mean(np.array(coords), axis=0)).tolist()
        neighbor_cutoff = 1.1 * a
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    coords_arr = np.array(coords, dtype=float)
    dist = distance_matrix(coords_arr, coords_arr)
    neighbors = [np.where((dist[i] < neighbor_cutoff) & (dist[i] > 1e-12))[0].tolist() for i in range(len(coords_arr))]
    return coords_arr, neighbors


def local_mu_values(coords: np.ndarray, mu0_over_u: float, onsite_u_j: float, mass_kg: float, trap_freq_hz: float):
    omega = 2 * np.pi * trap_freq_hz
    r2 = coords[:, 0] ** 2 + coords[:, 1] ** 2
    trap_energy_j = 0.5 * mass_kg * omega**2 * r2
    return mu0_over_u - trap_energy_j / onsite_u_j

@st.cache_data(show_spinner=False)
def run_site_dependent_gutzwiller(
    geometry: str,
    j_over_u: float,
    mu0_over_u: float,
    onsite_u_j: float,
    mass_kg: float,
    trap_freq_hz: float,
    lattice_spacing_m: float,
    lx: int,
    ly: int,
    nmax: int = 4,
):
    coords, neighbors = generate_lattice(geometry, lx, ly, lattice_spacing_m)
    mu_i = local_mu_values(coords, mu0_over_u, onsite_u_j, mass_kg, trap_freq_hz)

    num_sites = len(coords)
    psi = 0.05 * np.ones(num_sites, dtype=complex)
    density = np.zeros(num_sites)

    b, bdag, n_op = boson_operators(nmax)
    ident = np.eye(nmax + 1)
    n_int = n_op @ (n_op - ident)

    for _ in range(140):
        new_psi = np.zeros(num_sites, dtype=complex)
        new_density = np.zeros(num_sites)

        for i in range(num_sites):
            gamma_i = sum(psi[j] for j in neighbors[i])
            h_i = (
                -j_over_u * (np.conj(gamma_i) * b + gamma_i * bdag)
                + j_over_u * np.real(np.conj(psi[i]) * gamma_i) * ident
                + 0.5 * n_int
                - mu_i[i] * n_op
            )
            eigvals, eigvecs = la.eigh(h_i)
            gs = eigvecs[:, 0]
            new_psi[i] = gs.conj().T @ b @ gs
            new_density[i] = np.real(gs.conj().T @ n_op @ gs)

        if np.max(np.abs(new_psi - psi)) < 1e-6:
            density = new_density
            psi = new_psi
            break
        psi = 0.55 * new_psi + 0.45 * psi
        density = new_density

    return coords, density, np.abs(psi), psi, mu_i

# =============================================================================
# Correlations, TOF, exact diagonalization, and gap
# =============================================================================

@st.cache_data(show_spinner=False)
def calculate_tof(coords: np.ndarray, psi_complex: np.ndarray, density: np.ndarray):
    kx = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    ky = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    interference = np.zeros_like(kx_grid, dtype=complex)
    coords_scaled = coords / max(np.linalg.norm(coords, axis=1).max(), 1e-12) * 8
    for i in range(len(coords_scaled)):
        interference += psi_complex[i] * np.exp(1j * (kx_grid * coords_scaled[i, 0] + ky_grid * coords_scaled[i, 1]))

    coherent = np.abs(interference) ** 2
    incoherent_bg = np.sum(density) * np.exp(-(kx_grid**2 + ky_grid**2) / 10)
    wannier_envelope = np.exp(-(kx_grid**2 + ky_grid**2) / 15)
    return kx_grid, ky_grid, (coherent + incoherent_bg) * wannier_envelope

@st.cache_data(show_spinner=False)
def build_ed_hamiltonian(num_sites: int, bonds: tuple[tuple[int, int], ...], nmax: int, j: float, u: float, mu: float):
    basis = list(product(range(nmax + 1), repeat=num_sites))
    index = {state: i for i, state in enumerate(basis)}
    h_ed = np.zeros((len(basis), len(basis)))

    for state_idx, state_tuple in enumerate(basis):
        state = list(state_tuple)
        h_ed[state_idx, state_idx] += sum((u / 2) * ni * (ni - 1) - mu * ni for ni in state)

        for i, j_site in bonds:
            for source, target in [(i, j_site), (j_site, i)]:
                if state[source] > 0 and state[target] < nmax:
                    new_state = state.copy()
                    amp = -j * np.sqrt(state[source] * (state[target] + 1))
                    new_state[source] -= 1
                    new_state[target] += 1
                    h_ed[index[tuple(new_state)], state_idx] += amp
    return h_ed, basis

@st.cache_data(show_spinner=False)
def calculate_gap_curve(geometry: str):
    if geometry == "1D Chain":
        bonds = ((0, 1), (1, 2), (2, 3))
    elif geometry == "Triangular":
        bonds = ((0, 1), (1, 2), (0, 2), (1, 3), (2, 3))
    else:
        bonds = ((0, 1), (2, 3), (0, 2), (1, 3))

    j_vals = np.linspace(0, 0.15, 24)
    gaps = []
    for j in j_vals:
        h_ed, _ = build_ed_hamiltonian(4, bonds, 2, j, 1.0, 0.5)
        eigvals, _ = la.eigh(h_ed)
        gaps.append(eigvals[1] - eigvals[0])
    return j_vals, np.array(gaps)

# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(layout="wide", page_title="SF-MI Bose-Hubbard Explorer")
st.title("⚛️ Superfluid–Mott Insulator Bose-Hubbard Explorer")
st.caption("Interactive mean-field and exact-diagonalization tools for optical-lattice Bose-Hubbard physics.")

with st.sidebar:
    st.header("User Inputs")
    st.caption("Change parameters freely, then click **Run** to update the app.")

    with st.form("parameter_form"):
        species_input = st.selectbox("Atomic species", list(SPECIES_DATA.keys()))
        geometry_input = st.selectbox("Lattice geometry", list(GEOMETRIES.keys()), index=2)
        wavelength_nm_input = st.slider("Lattice wavelength (nm)", 500, 1500, 1064, step=1)
        lattice_spacing_nm_input = st.number_input(
            "Lattice constant / spacing a (nm)",
            min_value=100.0,
            max_value=2000.0,
            value=532.0,
            step=1.0,
            help="This controls the real-space site spacing used for the trap, density maps, coherence maps, and lattice geometry. For a standard counterpropagating lattice, a = wavelength/2.",
        )
        lattice_depth_er_input = st.slider("Lattice depth $V_0/E_R$", 2.0, 30.0, 14.0, step=0.5)
        mu0_over_u_input = st.slider("Central chemical potential $\\mu_0/U$", 0.0, 5.0, 1.7, step=0.05)
        trap_freq_hz_input = st.slider("Trap frequency (Hz)", 5, 200, 80, step=5)

        st.subheader("Lattice size")
        lx_input = st.slider("$L_x$", 7, 61, 31, step=2)
        ly_input = 1 if geometry_input == "1D Chain" else st.slider("$L_y$", 7, 61, 31, step=2)
        nmax_input = st.slider("Max occupancy per site $n_{max}$", 2, 7, 4, step=1)

        run_clicked = st.form_submit_button("Run", type="primary")

if run_clicked:
    st.session_state["last_inputs"] = {
        "species": species_input,
        "geometry": geometry_input,
        "wavelength_nm": wavelength_nm_input,
        "lattice_spacing_nm": lattice_spacing_nm_input,
        "lattice_depth_er": lattice_depth_er_input,
        "mu0_over_u": mu0_over_u_input,
        "trap_freq_hz": trap_freq_hz_input,
        "lx": lx_input,
        "ly": ly_input,
        "nmax": nmax_input,
    }

if "last_inputs" not in st.session_state:
    st.info("Choose parameters in the sidebar, then click **Run** to generate the phase diagrams and observables.")
    st.stop()

_inputs = st.session_state["last_inputs"]
species = _inputs["species"]
geometry = _inputs["geometry"]
wavelength_nm = _inputs["wavelength_nm"]
lattice_spacing_nm = _inputs.get("lattice_spacing_nm", wavelength_nm / 2)
lattice_spacing_m = lattice_spacing_nm * 1e-9
lattice_depth_er = _inputs["lattice_depth_er"]
mu0_over_u = _inputs["mu0_over_u"]
trap_freq_hz = _inputs["trap_freq_hz"]
lx = _inputs["lx"]
ly = _inputs["ly"]
nmax = _inputs["nmax"]

params = calculate_hubbard_params(species, wavelength_nm, lattice_depth_er)
z = GEOMETRIES[geometry]
phase_label = classify_point(params["J_over_U"], mu0_over_u, z)

with st.sidebar:
    st.markdown("---")
    st.subheader("Computed Bose-Hubbard Parameters")
    st.write(f"$E_R/h$ = {print_hz(params['E_R_J']) / 1e3:.3f} kHz")
    st.write(f"$J/h$ = {print_hz(params['J_J']):.3f} Hz")
    st.write(f"$U/h$ = {print_hz(params['U_J']):.3f} Hz")
    st.write(f"$J/U$ = {params['J_over_U']:.5f}")
    st.write(f"$U/J$ = {params['U_over_J']:.2f}")
    st.write(f"Lattice spacing $a$ = {lattice_spacing_nm:.1f} nm")
    st.info(f"Uniform MF classification: {phase_label}")

st.markdown(
    """
    This app maps experimental optical-lattice parameters onto the Bose-Hubbard model, then visualizes the SF-MI transition using uniform and site-dependent Gutzwiller mean-field calculations.
    """
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📍 Phase Diagram",
        "🔢 Uniform Mean Field",
        "🎂 Density / Wedding Cake",
        "🌌 Coherence + Momentum",
        "⚡ Excitation Gap",
    ]
)

# -----------------------------------------------------------------------------
# Tab 1: analytic phase diagram
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Analytic mean-field Mott lobes")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    #calculate the highest physical lobe needed for display
    max_lobe = int(np.ceil(mu0_over_u)) if mu0_over_u > 1 else 3
    max_lobe = min(max_lobe, nmax)
    
    for n in range(1, max_lobe + 1):
        mu_vals = np.linspace(n - 1 + 1e-4, n - 1e-4, 500)
        j_vals = analytic_mott_lobe(mu_vals, n, z)
        mask = j_vals > 0
        ax.plot(j_vals[mask], mu_vals[mask], lw=2, label=f"n={n} lobe")
    ax.scatter(params["J_over_U"], mu0_over_u, s=160, marker="*", edgecolor="black", label="current input")
    ax.set_xlim(0, 0.16)
    ax.set_ylim(-0.05, max_lobe + 0.3)
    ax.set_xlabel("$J/U$")
    ax.set_ylabel("$\\mu/U$")
    ax.set_title(f"{geometry} lattice, z={z}")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.write(
        "The star shows the current user-selected experimental point. The lobes are the analytic uniform mean-field prediction."
    )

# -----------------------------------------------------------------------------
# Tab 2: numerical uniform MFT
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Numerical uniform Gutzwiller mean-field")
    st.write("This solves the self-consistent single-site mean-field Hamiltonian on a grid and overlays the analytic lobe boundaries.")

    grid_mu_max = float(mu0_over_u + 0.5) if mu0_over_u > 1 else 3.3

    with st.spinner("Computing uniform mean-field grids..."):
        j_vals, mu_vals, psi_grid, density_grid = calc_uniform_mft_grids(z, grid_mu_max, nmax=nmax)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        c = ax.contourf(j_vals, mu_vals, psi_grid, levels=40, cmap="inferno")
        fig.colorbar(c, ax=ax, label=r"Order parameter $|\psi|$")
        max_lobe = int(np.ceil(mu0_over_u)) if mu0_over_u > 1 else 3
        max_lobe = min(max_lobe, nmax)
        for n in range(1, max_lobe + 1):
            mu_l = np.linspace(n - 1 + 1e-4, n - 1e-4, 250)
            j_l = analytic_mott_lobe(mu_l, n, z)
            ax.plot(j_l[j_l > 0], mu_l[j_l > 0], "w--", lw=1.8)
        ax.scatter(params["J_over_U"], mu0_over_u, s=120, marker="*", edgecolor="black", color="cyan")
        ax.set_xlabel("$J/U$")
        ax.set_ylabel("$\\mu/U$")
        ax.set_title("Numerical phase diagram")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        c = ax.imshow(
            density_grid,
            origin="lower",
            aspect="auto",
            extent=[j_vals[0], j_vals[-1], mu_vals[0], mu_vals[-1]],
            cmap="viridis",
        )
        fig.colorbar(c, ax=ax, label=r"Density $\langle n \rangle$")
        ax.scatter(params["J_over_U"], mu0_over_u, s=120, marker="*", edgecolor="black", color="red")
        ax.set_xlabel("$J/U$")
        ax.set_ylabel("$\\mu/U$")
        ax.set_title("Uniform mean-field density")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Tab 3: site-dependent density
# -----------------------------------------------------------------------------
with tab3:
    st.subheader(f"Site-dependent Gutzwiller calculation: {geometry}")
    st.write("The harmonic trap makes the local chemical potential spatially dependent, producing shell structure in the density.")

    with st.spinner("Solving site-dependent Gutzwiller equations..."):
        coords, density, abs_psi, psi_complex, mu_i = run_site_dependent_gutzwiller(
            geometry,
            params["J_over_U"],
            mu0_over_u,
            params["U_J"],
            params["mass_kg"],
            trap_freq_hz,
            lattice_spacing_m,
            lx,
            ly,
            nmax=nmax,
        )

    r_um = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) * 1e6
    x_um = coords[:, 0] * 1e6
    y_um = coords[:, 1] * 1e6
    lda = np.where(mu_i < 0, 0, np.floor(mu_i) + 1)
    lda = np.clip(lda, 0, nmax)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(r_um, density, s=13, alpha=0.65, label="Gutzwiller density")
        ax.scatter(r_um, lda, s=8, alpha=0.35, label="integer LDA check")
        ax.set_xlabel(r"Radius $r$ ($\mu$m)")
        ax.set_ylabel(r"Density $\langle n_i \rangle$")
        ax.set_title("Radial density profile")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(r_um, density, s=12, alpha=0.55, label=r"$\langle n_i\rangle$")
        ax.scatter(r_um, abs_psi, s=12, alpha=0.55, label=r"$|\psi_i|$")
        ax.set_xlabel(r"Radius $r$ ($\mu$m)")
        ax.set_title("Density and order parameter")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        if geometry == "1D Chain":
            sc = ax.scatter(x_um, density, c=density, s=50, cmap="viridis")
            ax.set_yticks([])
        else:
            sc = ax.scatter(x_um, y_um, c=density, s=34, cmap="viridis")
            ax.axis("equal")
        fig.colorbar(sc, ax=ax, label="Density")
        ax.set_title("Real-space density map")
        ax.set_xlabel(r"$x$ ($\mu$m)")
        ax.set_ylabel(r"$y$ ($\mu$m)")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        if geometry == "1D Chain":
            sc = ax.scatter(x_um, abs_psi, c=abs_psi, s=50, cmap="plasma")
            ax.set_yticks([])
        else:
            sc = ax.scatter(x_um, y_um, c=abs_psi, s=34, cmap="plasma")
            ax.axis("equal")
        fig.colorbar(sc, ax=ax, label=r"$|\psi_i|$")
        ax.set_title("Order parameter map")
        ax.set_xlabel(r"$x$ ($\mu$m)")
        ax.set_ylabel(r"$y$ ($\mu$m)")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Tab 4: coherence and TOF
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Mean-field coherence and simulated momentum distribution")
    with st.spinner("Computing real-space coherence and momentum distribution..."):
        coords, density, abs_psi, psi_complex, mu_i = run_site_dependent_gutzwiller(
            geometry,
            params["J_over_U"],
            mu0_over_u,
            params["U_J"],
            params["mass_kg"],
            trap_freq_hz,
            lattice_spacing_m,
            lx,
            ly,
            nmax=nmax,
        )
        kx_grid, ky_grid, n_k = calculate_tof(coords, psi_complex, density)

    r_um = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) * 1e6
    x_um = coords[:, 0] * 1e6
    y_um = coords[:, 1] * 1e6
    center_idx = int(np.argmin(coords[:, 0] ** 2 + coords[:, 1] ** 2))
    g1_raw = np.abs(np.conj(psi_complex[center_idx]) * psi_complex)
    g1_norm = g1_raw / np.sqrt((density[center_idx] + 1e-12) * (density + 1e-12))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        c = ax.contourf(kx_grid, ky_grid, n_k, levels=45, cmap="magma")
        fig.colorbar(c, ax=ax, label=r"$n(k)$")
        ax.set_title("Simulated time-of-flight / momentum image")
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        st.pyplot(fig)

        line_sites = np.where(np.abs(coords[:, 1] - coords[center_idx, 1]) < lattice_spacing_m / 2)[0]
        if len(line_sites) > 1:
            line_sites = line_sites[np.argsort(coords[line_sites, 0])]
            numerator = np.abs(np.outer(np.conj(psi_complex[line_sites]), psi_complex[line_sites]))
            denominator = np.sqrt(np.outer(density[line_sites] + 1e-12, density[line_sites] + 1e-12))
            g1_matrix = numerator / denominator
            fig, ax = plt.subplots(figsize=(6, 5))
            c = ax.imshow(g1_matrix, origin="lower", cmap="inferno", vmin=0, vmax=1)
            fig.colorbar(c, ax=ax, label=r"normalized $g^{(1)}(i,j)$")
            ax.set_title("Coherence matrix along center line")
            st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        if geometry == "1D Chain":
            sc = ax.scatter(x_um, g1_norm, c=g1_norm, s=50, cmap="magma")
            ax.set_yticks([])
        else:
            sc = ax.scatter(x_um, y_um, c=g1_norm, s=34, cmap="magma")
            ax.axis("equal")
        fig.colorbar(sc, ax=ax, label=r"normalized $g^{(1)}(i,0)$")
        ax.set_title("First-order coherence relative to center")
        ax.set_xlabel(r"$x$ ($\mu$m)")
        ax.set_ylabel(r"$y$ ($\mu$m)")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(r_um, abs_psi, s=12, alpha=0.5, label=r"$|\psi_i|$")
        ax.scatter(r_um, g1_raw, s=12, alpha=0.5, label=r"raw $|\psi_0^*\psi_i|$")
        ax.scatter(r_um, g1_norm, s=12, alpha=0.5, label=r"normalized $g^{(1)}$")
        ax.set_xlabel(r"Radius $r$ ($\mu$m)")
        ax.set_title("Coherence vs radius")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Tab 5: ED and excitation gap
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Finite-Size gap")


    j_gap, gaps = calculate_gap_curve(geometry)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(j_gap, gaps, "o-", color="black")
    ax.axvline(params["J_over_U"], color="red", ls="--", label="current $J/U$")
    ax.set_xlabel("$J/U$")
    ax.set_ylabel(r"Finite-size gap $\Delta E/U$")
    ax.set_title("Small-system excitation gap")
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Render the single remaining gap plot nicely in the center
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        st.pyplot(fig)

st.markdown("---")
st.caption("Built for Phys 452: Quantum Optics and Quantum Gases. Mean-field correlation plots use the Gutzwiller factorization ⟨b†ᵢbⱼ⟩ ≈ ψᵢ*ψⱼ.")