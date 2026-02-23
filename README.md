# Time-Reversal Refocusing in Homogeneous and Randomly Heterogeneous Media

Numerical study of time-reversal refocusing using the paraxial (Schrodinger) approximation. A Gaussian beam propagates through a medium, is captured by a time-reversal mirror (TRM) at `z = L`, and is back-propagated to `z = 2L`. In random media, the refocused profile exhibits super-resolution: a focal spot narrower than the homogeneous prediction.


## Overview

| Part | Topic | Method |
|------|-------|--------|
| 1 | Gaussian beam propagation (homogeneous) | Fourier method |
| 2 | Time reversal with compact and Gaussian mirrors | Fourier method |
| 3 | Propagation in a random medium | Split-step Fourier (Strang splitting) |
| 4 | Time reversal in a random medium | Split-step + Monte Carlo |
| 5 | Broadband time reversal (time-dependent) | Multi-frequency summation |


## Key Equations

**Paraxial (Schrodinger) equation** in a homogeneous medium:

$$\partial_z \phi = \frac{i}{2k} \partial_{xx} \phi, \qquad \phi(0, x) = e^{-x^2/r_0^2}$$

**Split-step extension** for a random medium with potential $\mu(z, x)$:

$$\partial_z \phi = \frac{i}{2k} \partial_{xx} \phi + \frac{ik}{2} \mu(z, x)\, \phi$$


## Project Structure

```
time-reversal-refocusing/
├── src/              # Solvers, random medium, time reversal
├── tests/            # pytest test suite
├── notebooks/        # Main notebook (deliverable)
├── scripts/          # Figure generation scripts
└── README.md
```


## Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| `r0` | 2 | Initial Gaussian beam radius |
| `N` | 2^10 = 1024 | Grid points |
| `x_max` | 60 | Domain `[-30, 30]` |
| `k = omega` | 1 | Wavenumber (c0 = 1) |
| `L` | 10 | Propagation distance |
| `h` | 1 | Longitudinal step size for split-step |
| `z_c` | 1 | Random medium correlation length (z) |
| `x_c` | 4 | Random medium correlation length (x) |
| `sigma` | 1 | Fluctuation amplitude |


## How to Run

```bash
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run the main notebook (primary deliverable)
jupyter notebook notebooks/notebook_main.ipynb
```

## References

- Fouque, Garnier, Papanicolaou, Solna. *Wave Propagation and Time Reversal in Randomly Layered Media*. Springer, 2007.
- Blomgren, Papanicolaou, Zhao. "Super-Resolution in Time-Reversal Acoustics." *JASA* 111 (2002): 230-248.
- Garnier, J. "Time-reversal refocusing for point source in randomly layered media." *Wave Motion* 42 (2005): 238-260.
- Fink, M. "Time reversed acoustics." *Physics Today* 50.3 (1997): 34-40.
- Garnier, J. *Inverse Problems* (lecture notes). MVA, ENS Paris-Saclay, 2025-2026.
