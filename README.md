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
    Auto-Sci explores scientific knowledge extracted from E3SM source code
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

### 2. Trace a scientific pathway

Auto-Sci identifies the pathway connecting water density, `rho_w`, and 
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
\rho_w =
\frac{
40g(\kappa z)^2 e^{-2\kappa_s z}
}{
w_s^2\left[(20Ri+1)^2-1\right]
}
\frac{\partial \rho_w}{\partial z}
```

Auto-Sci then compresses this multi-equation pathway into the following interpretable surrogate:

```math
\rho_w =
998.42
+ 0.15\log(\mathrm{savedtke1}+15.98)
+ \frac{16.89}{\mathrm{savedtke1}+15.98}
```

This relationship indicates that stored turbulent kinetic energy (`savedtke1`) influences water density through coupled turbulence, stability, and stratification processes.

### 2. Soil temperature and methane concentration

The full mechanistic relationship is:

```math
t_{\mathrm{soisno}} =
Q_b
+ \frac{10}{\ln Q_m}
\ln\left[
\frac{
\epsilon C_t-\mathcal{D}_z(C)+\mathcal{L}_{CH_4}
}{
f_m B\phi(z)/\Delta z
}
\right]
```

Auto-Sci derives the following compact empirical surrogate:

```math
t_{\mathrm{soisno}} =
14.27
+ 0.9363\ln\left(C+2.1\times10^{-4}\right)
- \frac{0.00010367}{C+2.1\times10^{-4}}
```

This relationship links soil temperature (`t_soisno`) with methane concentration (\(C\)) through methane production, transport, and loss processes.

For both case studies, the surrogate predictions are compared with the corresponding E3SM simulation outputs to evaluate predictive accuracy and scientific consistency.



---

## Repository Contents

| Directory | Description | Availability |
|---|---|---|
| `data/e3sm_fortran/` | E3SM Fortran files used as Auto-Sci inputs | Available |
| `data/e3sm_validation/` | E3SM numerical data used for validation | Available |
| `data/science_graph/` | Extracted science-graph nodes and edges | Available |
| `results/equation_discovery/` | Discovered mathematical relationships | Available |
| `results/equation_validation/` | Predictions, residuals, and error measurements | Available |
| `results/supplementary_tables/` | Supporting numerical result tables | Available |
| `figures/` | Framework, graph, pathway, and validation figures | Available |
| `src/` | Complete Auto-Sci implementation | **Coming soon** |

## Data

The research data supporting Auto-Sci are organized into three groups.

<details>
<summary><strong>E3SM Fortran input files</strong></summary>

<br>

Location:

```text
data/e3sm_fortran/
