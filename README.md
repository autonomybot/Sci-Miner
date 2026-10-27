<div align="center">

# Sci-Miner

### Sci-Miner: AI-Enabled Scientific Mining in Complex Computational Models

**Sci-Miner transforms scientific source code into a connected science graph  
for relationship exploration, equation discovery, and numerical validation.**

<br>

[![Data](https://img.shields.io/badge/Data-Available-2E8B57?style=flat-square)](#data)
[![Code](https://img.shields.io/badge/Source%20Code-Coming%20Soon-F39C12?style=flat-square)](#source-code)
[![Model](https://img.shields.io/badge/Scientific%20Model-E3SM-4169E1?style=flat-square)](https://e3sm.org/)
[![Graph](https://img.shields.io/badge/Science%20Graph-Neo4j-018BFF?style=flat-square)](https://neo4j.com/)

</div>

---

<p align="center">
  <img src="figures/auto_sci_demo.gif" width="900" alt="Auto-Sci demonstration">
</p>

<p align="center">
  <em>
    Sci-Miner explores scientific knowledge extracted from E3SM source code
    and identifies testable relationships between scientific quantities.
  </em>
</p>

## Overview

Large scientific computational models contain extensive scientific knowledge. 
However, this knowledge is distributed across thousands of source files, 
variables, equations, and model components.

**Sci-Miner** converts scientific source code into a structured science graph. 
The framework allows researchers to:

- explore scientific variables and equations;
- visualize connections across model components;
- trace pathways between selected scientific quantities;
- derive interpretable mathematical relationships; and
- validate discovered relationships using E3SM numerical data.

<!-- <p align="center">
  <img src="figures/graphical_abstract.png" width="900" alt="Auto-Sci graphical abstract">
</p> -->

## Demonstrated Capabilities

### 1. Explore the science graph

Auto-Sci organizes variables, equations, files, and their dependencies into a 
connected scientific representation.

<p align="center">
  <img src="figures/Figure 02.png" width="850" alt="Science graph">
</p>

[Click the preview to open the interactive science graph in Cosmograph.](https://cosmograph.app/run/?data=https://https://raw.githubusercontent.com/autonomybot/Sci-Miner/refs/heads/main/data/output/e3sm.csv)

### 2. Trace a scientific pathway

Sci-Miner identifies the pathway connecting water density, `rho_w`, and 
turbulent kinetic energy, `savedtke1`.

<p align="center">
  <img src="figures/Figure 06.png" width="850"
       alt="Pathway connecting different variables">
</p>

### 3. Discover and validate relationships

## Scientific Relationships Discovered by Auto-Sci

Auto-Sci identified two cross-module scientific relationships. For each relationship, the framework reconstructs the full mechanistic dependency and derives a compact symbolic surrogate.

### 1. Lake water density and turbulent kinetic energy

The full mechanistic relationship is:

```math
\rho_w = \frac{g\,(\kappa z)^2\, e^{-2k_s z}}{w_s^2\;\mathcal{R}i\,\left(10\,\mathcal{R}i + 1\right)}\cdot\frac{\partial \rho_w}{\partial z},
    \qquad
    \mathcal{R}i^2 = \frac{1}{37}\left(\frac{\kappa\, w_s\, z\, c_{wat}\, e^{-k_s z}}{\mathrm{savedtke}_1 - k_m\, c_{wat}} - 1\right).
```

Auto-Sci then compresses this multi-equation pathway into the following interpretable surrogate:

```math
\rho_w=\frac{263.8000434}{R\,[10R+1]}, \quad R=\sqrt{\frac{1}{37}\left(\frac{1.100375669\times10^{10}}{\mathrm{savedtke}_1+7180533846}-1\right)}
```

This relationship indicates that stored turbulent kinetic energy (`savedtke1`) influences water density through coupled turbulence, stability, and stratification processes.

### 2. Soil temperature and methane concentration

The full mechanistic relationship is:

```math
t_{\mathrm{soisno}} = Q_{10,\mathrm{base}} + \frac{10}{\ln Q_{10,\mathrm{CH_4}}}
    \ln\!\left[ \frac{\epsilon \dfrac{\partial C}{\partial t} - \dfrac{\partial}{\partial z}\!\left(D\dfrac{\partial C}{\partial z}\right) + \mathcal{L}_{\mathrm{CH_4}}}
    {f_{\mathrm{CH_4}}\; \mathrm{base_{decomp}}\; \phi(z)/dz} \right]
```

Auto-Sci derives the following compact empirical surrogate:

```math
t_{\mathrm{soisno}}[\mathrm{K}] = 262.221+16.36611\left(\frac{C}{10^{-4}\,\mathrm{mol\,m^{-3}}}\right)-2.41791\left(\frac{C}{10^{-4}\,\mathrm{mol\,m^{-3}}}\right)^2 .
    \label{eq:methane-temperature-surrogate}
```

This relationship links soil temperature (`t_soisno`) with methane concentration (\(C\)) through methane production, transport, and loss processes.

For both case studies, the surrogate predictions are compared with the corresponding E3SM simulation outputs to evaluate predictive accuracy and scientific consistency.



---

## Repository Contents

| Directory | Description | Availability |
|---|---|---|
| `data/input/fortran/` | E3SM Fortran source files used as inputs | Available |
| `data/validation/` | E3SM numerical data used for validation | Available |
| `data/output/` | Extracted science-graph nodes and edges | Available |
| `graph_generator/` | Science-graph extraction and conversion tools | Available |
| `path_mining_and_validation/` | Path mining, provenance checks, and numerical replay | Available |
| `function_approximation/` | FEX-based equation discovery and surrogate fitting | Available |
| `comparision/` | FEX, SINDy, neural-network, and path-recovery baselines | Available |
| `e3sm_validation/` | Fixed-equation validation diagnostics | Available |
| `results/` | Saved metrics, predictions, and supplementary tables | Available |
| `figures/` | Framework, graph, pathway, and validation figures | Available |
| `app/` | Interactive science-graph visualization and LLM-based questions | **Coming soon** |


<details>
<summary><strong>E3SM Fortran input files</strong></summary>

<br>

Location:

```text
data/e3sm_fortran/
