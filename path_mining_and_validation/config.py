"""Central settings for path mining and validation.

Everything tunable lives here (paths, score weights, thresholds, case-study
queries) so the numbers in the paper can be regenerated and audited from one
place. Paths can be overridden with environment variables of the same name.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------- inputs
EDGES_CSV = os.getenv("EDGES_CSV", os.path.join(ROOT, "data", "output", "e3sm.csv"))
FORTRAN_DIR = os.getenv("FORTRAN_DIR", os.path.join(ROOT, "data", "input", "fortran"))
VALIDATION_DIR = os.getenv("VALIDATION_DIR", os.path.join(ROOT, "data", "validation"))
LAKE_CSV = os.path.join(VALIDATION_DIR, "Hourly lake-physics output from the E3SM lake-site simulation.csv")
LAND_CSV = os.path.join(VALIDATION_DIR, "Reduced_land-model_output_lake-site_simulation.csv")

# ---------------------------------------------------------------- outputs
FIG_DIR = os.getenv("FIG_DIR", os.path.join(ROOT, "figures"))
RESULT_DIR = os.getenv("RESULT_DIR", os.path.join(ROOT, "results", "path_mining_validation"))
SEED = 0

# ------------------------------------------------------- graph cleaning
# Tokens the LLM extraction returned as "variables" that are LaTeX/markup or
# language keywords, not scientific quantities.
LATEX_STOPWORDS = {
    "frac", "left", "right", "cdot", "times", "partial", "sum", "int", "sqrt", "exp", "ln", "log",
    "text", "mathrm", "mathbf", "mathit", "boxed", "oxed", "begin", "end", "quad", "qquad", "min",
    "max", "abs", "sin", "cos", "tan", "tanh", "if", "else", "then", "do", "mod", "and", "or", "not",
    "infty", "approx", "leq", "geq", "neq", "le", "ge", "to", "in", "array", "cases", "tag", "label",
    "dots", "ldots", "cdots", "sign", "nint", "real", "double", "precision", "integer", "dd",
}
# Loop / index / single-letter symbols connect a huge share of all equations,
# so they are never allowed as *interior* nodes of a path (the paper calls these
# "indexing variables" and generic hubs).
INDEX_SYMBOLS = {"i", "j", "k", "l", "m", "n", "c", "p", "g", "t", "d", "e", "f", "s", "r", "u", "v", "w", "x", "y", "z"}

# ------------------------------------------------------------ path mining
K_CANDIDATES = 10          # loop-free candidate paths generated per query (Yen's algorithm)
HUB_EDGE_WEIGHT = 0.5      # candidate generation avoids hubs: w(u,v) = 1 + HUB*(log1p(deg u)+log1p(deg v))/2
MAX_EQUATIONS = 8          # discard paths with more equations than this
# Eq.(5):  S(P) = (ALPHA*V(P) + BETA*Spec(P) + GAMMA*Phys(P)) / (1 + LAMBDA*|P|)
ALPHA, BETA, GAMMA, LAMBDA = 1.0, 0.5, 0.5, 0.15
# Eq.(7):  V = w_src*S_src + w_var*S_var + w_sym*S_sym + w_num*S_num
# (weights of unavailable components are dropped and the rest renormalised)
W_SRC, W_VAR, W_SYM, W_NUM = 0.25, 0.25, 0.25, 0.25
VERIFIED_MIN = 0.80        # V >= this and every available component >= COMPONENT_MIN
COMPONENT_MIN = 0.50
PARTIAL_MIN = 0.50
SRC_TOKEN_MATCH = 0.50     # an equation is "source supported" if >= this share of its identifiers is in its file

# ------------------------------------------------------------ experiments
N_RANDOM_PAIRS = 150       # random variable pairs for the benchmark
MIN_SYMBOL_DEGREE = 3      # endpoints of random pairs need at least this many equations
SYMBOLIC_SAMPLE = 1500     # equations sampled for the graph-wide symbolic pass rate

# Case studies from the paper: (label, source symbol, target symbol). Symbols
# are matched on their normalised name; unresolved ones are reported, never faked.
CASE_STUDIES = [
    ("Lake chain (Fortran)", "rhow", "savedtke1"),
    ("Lake chain: density to Richardson", "rhow", "ri"),
    ("Lake chain: temperature to buoyancy", "t_lake", "n2"),
    ("Fig.5A lake pathway", "rho_w", "savedtke1"),
    ("Fig.4 aerosol pathway", "NH3", "dT/dy"),
    ("Fig.5B methane pathway", "T_soisno", "conc_ch4_sat"),
    ("Density to Richardson number", "rho_w", "Ri"),
    ("Mixing energy to buoyancy", "tke", "N2"),
    ("Soil temperature to methane", "T_Soisno", "CH4"),
    ("von Karman to Richardson number", "kappa", "Ri"),
]
