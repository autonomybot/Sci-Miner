"""Supervised-data adapter for the supplied Finite Expression Method tree.

The archive supplied with the project is a PDE solver, so its controller
optimizes a PDE residual rather than a CSV regression loss.  This adapter keeps
the archive's ``depth2`` tree and operator grammar exactly, replacing only the
PDE residual with the supervised E3SM mean-squared error.  The provenance is
explicit in the output; this is not presented as a run of the archive's PDE
entry point.

Archive grammar copied from ``fex/*/function.py`` and
``fex/*/controller_*.py``:

* unary: ``0, 1, x, x^2, x^3, x^4, exp, sin, cos``;
* binary: ``+, *, -``;
* ``depth2``: two binary subtrees, each wrapped by an affine unary node,
  joined by a binary root; each leaf applies a unary operator to each input
  before its trainable affine map.

Run from the repository root::

    python function_approximation/fex_original_regression.py

The script writes ``fex_original_lake_methane_equations.txt``, machine
readable metadata, and ten held-out rows for each case.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_TXT = HERE / "fex_original_lake_methane_equations.txt"
OUT_JSON = HERE / "fex_original_lake_methane_metadata.json"
CWAT = 4.188e6
SEED = 0
TRAIN_FRACTION = 0.70
SEARCH_CANDIDATES = 192
SEARCH_STEPS = 120
FINETUNE_STEPS = 400
SEARCH_ROWS = 2000

torch.set_num_threads(2)

# Exact order in the supplied archive's function.py.
UNARY = ("0", "1", "id", "sq", "cube", "quartic", "exp", "sin", "cos")
BINARY = ("+", "*", "-")


def uop(x: torch.Tensor, op: int) -> torch.Tensor:
    if op == 0:
        return torch.zeros_like(x)
    if op == 1:
        return torch.ones_like(x)
    if op == 2:
        return x
    if op == 3:
        return x**2
    if op == 4:
        return x**3
    if op == 5:
        return x**4
    # The clamp is only a numerical guard for exp; the selected expression is
    # reported as exp in the archive grammar and remains deterministic.
    if op == 6:
        return torch.exp(torch.clamp(x, -20.0, 20.0))
    if op == 7:
        return torch.sin(x)
    if op == 8:
        return torch.cos(x)
    raise ValueError(op)


def bop(a: torch.Tensor, b: torch.Tensor, op: int) -> torch.Tensor:
    if op == 0:
        return a + b
    if op == 1:
        return a * b
    if op == 2:
        return a - b
    raise ValueError(op)


def forward(params: dict[str, torch.Tensor], x: torch.Tensor, choices: dict[str, list[int] | int]) -> torch.Tensor:
    """Evaluate the archive depth2 tree on standardized inputs."""
    leaf_w, leaf_b = params["leaf_w"], params["leaf_b"]
    inner_a, inner_b = params["inner_a"], params["inner_b"]
    # The original learnable tree applies the leaf unary operation elementwise,
    # then its trainable Linear(dim, 1) map.
    leaf = []
    for i in range(4):
        leaf_value = uop(x, choices["leaf_unary"][i])
        leaf.append((leaf_value * leaf_w[i]).sum(dim=1) + leaf_b[i])
    left_binary = bop(leaf[0], leaf[1], choices["inner_binary"][0])
    right_binary = bop(leaf[2], leaf[3], choices["inner_binary"][1])
    left = inner_a[0] * uop(left_binary, choices["inner_unary"][0]) + inner_b[0]
    right = inner_a[1] * uop(right_binary, choices["inner_unary"][1]) + inner_b[1]
    return bop(left, right, choices["root_binary"])


def init_params(dim: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))
    params = {
        "leaf_w": torch.randn((4, dim), generator=gen, dtype=torch.float64) / max(dim, 1) ** 0.5,
        "leaf_b": torch.zeros(4, dtype=torch.float64),
        "inner_a": torch.ones(2, dtype=torch.float64),
        "inner_b": torch.zeros(2, dtype=torch.float64),
    }
    for p in params.values():
        p.requires_grad_(True)
    return params


def train_candidate(x: torch.Tensor, y: torch.Tensor, choices: dict, rng: np.random.Generator,
                    steps: int, lr: float) -> tuple[dict[str, torch.Tensor], float]:
    params = init_params(x.shape[1], rng)
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    best = float("inf")
    best_state = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = forward(params, x, choices)
        loss = torch.mean((pred - y) ** 2)
        if not torch.isfinite(loss):
            return params, float("inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 10.0)
        opt.step()
        value = float(loss.detach().cpu().item())
        if value < best:
            best = value
            best_state = {k: v.detach().clone() for k, v in params.items()}
    if best_state is not None:
        for k, v in params.items():
            v.data.copy_(best_state[k])
    return params, best


def sample_choices(rng: np.random.Generator) -> dict:
    return {
        "leaf_unary": rng.integers(0, len(UNARY), size=4).tolist(),
        "inner_binary": rng.integers(0, len(BINARY), size=2).tolist(),
        "inner_unary": rng.integers(0, len(UNARY), size=2).tolist(),
        "root_binary": int(rng.integers(0, len(BINARY))),
    }


def fit_fex(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    cut = int(TRAIN_FRACTION * len(y))
    train, test = np.sort(order[:cut]), np.sort(order[cut:])
    x_mu, x_sd = x[train].mean(axis=0), x[train].std(axis=0)
    x_sd = np.where(x_sd == 0, 1.0, x_sd)
    y_mu, y_sd = float(y[train].mean()), float(y[train].std() or 1.0)
    z = (x - x_mu) / x_sd
    t = (y - y_mu) / y_sd
    search = np.sort(rng.choice(train, min(SEARCH_ROWS, len(train)), replace=False))
    xt = torch.tensor(z[search], dtype=torch.float64)
    yt = torch.tensor(t[search], dtype=torch.float64)
    best = None
    for idx in range(SEARCH_CANDIDATES):
        choices = sample_choices(rng)
        params, loss = train_candidate(xt, yt, choices, rng, SEARCH_STEPS, 0.03)
        if best is None or loss < best[0]:
            best = (loss, choices, params)
    assert best is not None
    choices = best[1]
    # Refit the selected symbolic tree on all training rows with the same
    # grammar, which mirrors the archive's final finetuning stage.
    xf = torch.tensor(z[train], dtype=torch.float64)
    yf = torch.tensor(t[train], dtype=torch.float64)
    params, train_loss = train_candidate(xf, yf, choices, rng, FINETUNE_STEPS, 0.01)
    with torch.no_grad():
        pred_norm = forward(params, torch.tensor(z, dtype=torch.float64), choices).cpu().numpy()
    pred = y_mu + y_sd * pred_norm
    scale = {"x_mean": x_mu.tolist(), "x_scale": x_sd.tolist(), "y_mean": y_mu, "y_scale": y_sd}
    return choices, params, pred, train, test, scale


def fmt(x: float) -> str:
    if abs(x) < 1e-12:
        x = 0.0
    return f"{float(x):.8g}"


def leaf_expr(params: dict[str, torch.Tensor], leaf_idx: int, op: int, names: list[str]) -> str:
    """Render the archive's affine-of-unary leaf with unambiguous signs."""
    weights = [float(v.detach().cpu().item()) for v in params["leaf_w"][leaf_idx]]
    bias = float(params["leaf_b"][leaf_idx].detach().cpu().item())
    if op == 0:
        return fmt(bias)
    if op == 1:
        return fmt(sum(weights) + bias)
    atoms = {2: "u_{{{}}}", 3: "\\left(u_{{{}}}\\right)^2", 4: "\\left(u_{{{}}}\\right)^3", 5: "\\left(u_{{{}}}\\right)^4",
             6: "\\exp\\!\\left(u_{{{}}}\\right)", 7: "\\sin\\!\\left(u_{{{}}}\\right)", 8: "\\cos\\!\\left(u_{{{}}}\\right)"}
    pieces = []
    for j, w in enumerate(weights):
        if abs(w) < 1e-11:
            continue
        term = f"{fmt(abs(w))}\\,{atoms[op].format(j + 1)}"
        pieces.append(("+" if w >= 0 else "-", term))
    if abs(bias) >= 1e-11 or not pieces:
        pieces.append(("+" if bias >= 0 else "-", fmt(abs(bias))))
    first_sign, first_term = pieces[0]
    text = ("-" if first_sign == "-" else "") + first_term
    for sign, term in pieces[1:]:
        text += f" {sign} {term}"
    return text


def apply_unary(op: int, text: str) -> str:
    if op == 0:
        return "0"
    if op == 1:
        return "1"
    if op == 2:
        return text
    if op == 3:
        return f"\\left({text}\\right)^2"
    if op == 4:
        return f"\\left({text}\\right)^3"
    if op == 5:
        return f"\\left({text}\\right)^4"
    return f"{unary_name(op)}\\!\\left({text}\\right)"


def unary_name(op: int) -> str:
    return {6: "\\exp", 7: "\\sin", 8: "\\cos"}[op]


def binary_name(op: int) -> str:
    return {0: "+", 1: "\\times", 2: "-"}[op]


def combine(left: str, op: int, right: str) -> str:
    return f"\\left({left}\\right) {binary_name(op)} \\left({right}\\right)"


def sympy_unary(value: sp.Expr, op: int) -> sp.Expr:
    if op == 0:
        return sp.Integer(0)
    if op == 1:
        return sp.Integer(1)
    if op == 2:
        return value
    if op == 3:
        return value**2
    if op == 4:
        return value**3
    if op == 5:
        return value**4
    if op == 6:
        return sp.exp(value)
    if op == 7:
        return sp.sin(value)
    if op == 8:
        return sp.cos(value)
    raise ValueError(op)


def sympy_binary(left: sp.Expr, right: sp.Expr, op: int) -> sp.Expr:
    return (left + right, left * right, left - right)[op]


def expression_latex(choices: dict, params: dict[str, torch.Tensor], names: list[str]) -> str:
    # Construct the same tree symbolically so constants and zero leaves from
    # the finite grammar are removed before the equation is placed in a paper.
    u = sp.symbols("u_1:" + str(len(names) + 1), real=True)
    weights = params["leaf_w"].detach().cpu().numpy()
    biases = params["leaf_b"].detach().cpu().numpy()
    leaves = []
    for i, op in enumerate(choices["leaf_unary"]):
        value = sum(sp.Float(float(weights[i, j]), 9) * sympy_unary(u[j], op) for j in range(len(names)))
        leaves.append(value + sp.Float(float(biases[i]), 9))
    left_bin = sympy_binary(leaves[0], leaves[1], choices["inner_binary"][0])
    right_bin = sympy_binary(leaves[2], leaves[3], choices["inner_binary"][1])
    left = sp.Float(float(params["inner_a"][0]), 9) * sympy_unary(left_bin, choices["inner_unary"][0]) + sp.Float(float(params["inner_b"][0]), 9)
    right = sp.Float(float(params["inner_a"][1]), 9) * sympy_unary(right_bin, choices["inner_unary"][1]) + sp.Float(float(params["inner_b"][1]), 9)
    expression = sympy_binary(left, right, choices["root_binary"])
    # One-input expressions simplify quickly and benefit from removing dead
    # zero/one leaves.  Multivariate trigonometric trees can make a full
    # symbolic simplifier spend minutes; their affine constants are already
    # reduced by SymPy construction, so retain the exact tree there.
    if len(names) == 1:
        expression = sp.simplify(expression)
    return sp.latex(expression, mul_symbol=r"\,")


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p - y
    sst = np.sum((y - y.mean()) ** 2)
    rmse = float(np.sqrt(np.mean(e**2)))
    return {"n": int(len(y)), "r2": float(1 - np.sum(e**2) / sst), "rmse": rmse,
            "mae": float(np.mean(np.abs(e))), "nrmse": rmse / float(np.ptp(y)),
            "median_abs_error": float(np.median(np.abs(e))), "max_abs_error": float(np.max(np.abs(e)))}


def load_case(case: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    import sys
    sys.path.insert(0, str(ROOT / "comparision"))
    import common
    data = common.CASES[case]["load"]().copy()
    if case == "lake":
        data["savedtke1"] = data["kme1"] * CWAT
        return data, data[["savedtke1"]].to_numpy(float), data["rho_w"].to_numpy(float), ["savedtke1"]
    data = data.rename(columns={"conc1": "conc_CH4_sat_col"})
    return data, data[["conc_CH4_sat_col"]].to_numpy(float), data["t_soisno"].to_numpy(float), ["conc_CH4_sat_col"]


def write_table(key: str, case: str, data: pd.DataFrame, x: np.ndarray, y: np.ndarray, p: np.ndarray, test: np.ndarray) -> None:
    chosen = np.sort(test)[:10]
    rows = []
    for i in chosen:
        site = int(data.iloc[i]["site"])
        rid = ("L" if case == "lake" else "M") + f"-g{site}-r{i+1:05d}"
        err = p[i] - y[i]
        rows.append(dict(record_id=rid, source_row=int(i + 1), site=site, predictor=x[i, 0], observed=y[i],
                         predicted=p[i], error=err, absolute_error=abs(err),
                         relative_error_percent=100 * abs(err) / max(abs(y[i]), 1e-12), split="held_out"))
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / f"fex_original_{key}_table.csv", index=False)
    lines = [f"    {r.record_id} & ${r.predictor:.6e}$ & {r.observed:.6f} & {r.predicted:.6f} & {r.absolute_error:.6f} & {r.relative_error_percent:.6f} \\\\" for r in frame.itertuples()]
    (HERE / f"fex_original_{key}_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = [
        "AI-Miner original-FEX-grammar supervised regression",
        "====================================================",
        "Source archive: Finite-expression-method-main.zip (fex/*/function.py and controller_*).",
        "Adaptation: the archive's PDE residual is replaced by E3SM supervised MSE; tree/operators are unchanged.",
        f"depth2 tree; seed={SEED}; random split={TRAIN_FRACTION:.0%}/{1-TRAIN_FRACTION:.0%}; candidates={SEARCH_CANDIDATES}; search steps={SEARCH_STEPS}; finetune steps={FINETUNE_STEPS}; search rows={SEARCH_ROWS}.",
        "",
    ]
    metadata = {"seed": SEED, "train_fraction": TRAIN_FRACTION, "tree": "depth2", "unary": list(UNARY), "binary": list(BINARY), "cases": {}}
    specs = [
        ("lake_savedtke1", "lake", ["savedtke1"]),
        ("lake_pathway", "lake", ["tke1", "savedtke1", "drhodz1", "ri1", "ws", "ks"]),
        ("methane_concentration", "methane", ["conc_CH4_sat_col"]),
    ]
    for key, case, names in specs:
        data, _, y, _ = load_case(case)
        x = data[names].to_numpy(float)
        choices, params, pred, train, test, scale = fit_fex(x, y, SEED)
        expr = expression_latex(choices, params, names)
        target = r"\rho_w" if case == "lake" else r"t_{\mathrm{soisno}}"
        labels = ([r"\mathrm{savedtke}_{1}"] if key == "lake_savedtke1" else
                  [r"TKE_1", r"\mathrm{savedtke}_{1}", r"\partial_z\rho_w", r"Ri_1", r"w_s", r"k_s"] if key == "lake_pathway" else
                  [r"\mathrm{conc}_{\mathrm{CH_4,sat,col}}"])
        u_defs = [f"u_{{{j+1}}}=({labels[j]}-({fmt(scale['x_mean'][j])}))/({fmt(scale['x_scale'][j])})" for j in range(len(names))]
        x_label = ", ".join(labels)
        eq = f"{target}=({fmt(scale['y_mean'])})+({fmt(scale['y_scale'])})\\,[{expr}]"
        m = {"all": metric(y, pred), "train": metric(y[train], pred[train]), "held_out": metric(y[test], pred[test])}
        pjson = {k: v.detach().cpu().numpy().tolist() for k, v in params.items()}
        metadata["cases"][key] = {"input": names, "target": target, "input_standardization": u_defs, "input_mean": scale["x_mean"], "input_scale": scale["x_scale"], "target_mean": scale["y_mean"], "target_scale": scale["y_scale"], "expression": expr, "equation": eq, "choices": choices, "parameters": pjson, "metrics": m}
        write_table(key, case, data, x, y, pred, test)
        report += [key.upper(), "-" * len(key), f"records: {len(data):,}; input: {x_label}; target: {target}", "standardized inputs: " + "; ".join(u_defs), "normalized expression:", expr, "dimensional equation:", eq, "choices: " + json.dumps(choices, sort_keys=True), "metrics: " + json.dumps(m, sort_keys=True), ""]
    OUT_TXT.write_text("\n".join(report) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TXT}")
    print(f"wrote {OUT_JSON}")
    for case, item in metadata["cases"].items():
        print(f"{case}: {item['equation']}")
        print(f"  held-out: {item['metrics']['held_out']}")


if __name__ == "__main__":
    main()
