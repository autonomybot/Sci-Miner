"""Step 2: parse math.txt (written by math_txt_generator.py) into an edge list
CSV (Source, Target, Relationship) usable by Cytoscape and sci-kg_contructor.py."""

import csv
import re

from api_config import EDGES_CSV_PATH, MATH_TXT_PATH, ROOT_NODE_NAME


def section_lines(block, header_pattern):
    """Return the cleaned lines of every '### <header>' section in a file block."""
    lines = []
    for m in re.finditer(rf"###\s*{header_pattern}\s*:(.*?)(?=###|={{5,}}|\Z)", block, re.DOTALL | re.IGNORECASE):
        for line in m.group(1).splitlines():
            # drop list bullets / markdown emphasis the LLM may add
            line = re.sub(r"^[\s\-*•\d.]+", "", line).replace("**", "").strip()
            if line and line != "...":
                lines.append(line)
    return lines


def parse_entities(lines):
    """'Name (symbol)' -> (name, symbol); a line without parentheses is its own symbol."""
    for line in lines:
        match = re.match(r"(.+?)\s*\((.+)\)\s*$", line)
        if match:
            yield match.group(1).strip(), match.group(2).strip()
        else:
            yield line, line


def symbol_in(symbol, text):
    """True if symbol appears as a whole token (not part of a longer name or a \latex command)."""
    # ``\w`` must be interpreted by the regex engine.  Using ``[\\w]`` here
    # would only exclude a literal backslash or the letter ``w`` and would
    # incorrectly match a symbol inside a longer identifier (e.g. ``abc`` in
    # ``xabc``).
    return re.search(rf"(?<!\w){re.escape(symbol)}(?!\w)", text) is not None


def main():
    with open(MATH_TXT_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    file_blocks = re.split(r"===== File: (.*?) =====", content)[1:]
    edges = []
    # Symbols are global graph identities: equal names across files are merged,
    # without resolving Fortran module or subroutine scope.
    all_variables = {}  # symbol -> (name, source_file)
    all_constants = {}  # symbol -> (name, source_file)
    all_equations = []  # (equation_text, source_file)

    total_files = len(file_blocks) // 2
    print(f"Detected {total_files} Fortran files for processing.")

    # First pass: extract entities
    for i in range(0, len(file_blocks), 2):
        filename = file_blocks[i].strip()
        block = file_blocks[i + 1]
        print(f"Processing file: {filename} ({i // 2 + 1}/{total_files})")

        # Prompt asks for "Global Variables" and "Local Variables"; plain "Variables" also accepted.
        for name, symbol in parse_entities(section_lines(block, r"(?:Global\s+|Local\s+)?Variables")):
            all_variables[symbol] = (name, filename)
            edges.append((filename, symbol, "contains"))

        for name, symbol in parse_entities(section_lines(block, "Constants")):
            all_constants[symbol] = (name, filename)
            edges.append((filename, symbol, "contains"))

        for eqn in section_lines(block, "Equations"):
            if "=" in eqn:
                all_equations.append((eqn, filename))
                edges.append((filename, eqn, "encodes"))

    print("Finished entity extraction.")
    print(f"  Variables: {len(all_variables)}")
    print(f"  Constants: {len(all_constants)}")
    print(f"  Equations: {len(all_equations)}")

    # Infer constants from equations: upper-case / Greek symbols not already known
    # This is a naming heuristic, not a check for a Fortran PARAMETER declaration.
    for eqn_text, _ in all_equations:
        for symbol in re.findall(r"\b([A-ZΔΩΓθμρσλφψζτνπ]{1,5})\b", eqn_text):
            if symbol not in all_variables and symbol not in all_constants:
                all_constants[symbol] = ("(inferred)", "inferred_file")
                edges.append(("inferred_file", symbol, "contains"))

    # Link variables/constants to equations (cross-file). Each equation is tokenised once and
    # looked up in a symbol index, instead of running one regex per symbol per equation.
    var_by_symbol = {sym: sym for sym in all_variables}
    const_by_symbol = {sym: sym for sym in all_constants}
    for idx, (eqn_text, _) in enumerate(all_equations):
        if idx % max(1, len(all_equations) // 10) == 0:
            print(f"Linking progress: {idx / len(all_equations) * 100:.1f}%")

        tokens = set(re.findall(r"(?<!\w)\w+", eqn_text))
        for symbol in tokens & var_by_symbol.keys():
            edges.append((symbol, eqn_text, "hasVariable"))
        for symbol in tokens & const_by_symbol.keys():
            edges.append((symbol, eqn_text, "hasConstant"))

    # Root node -> every Fortran file
    file_nodes = {n for src, tgt, _ in edges for n in (src, tgt) if n.lower().endswith(".f90")}
    for file_node in sorted(file_nodes):
        edges.append((ROOT_NODE_NAME, file_node, "has"))

    # Remove duplicate edges but keep order
    edges = list(dict.fromkeys(edges))

    # The output directory must already exist; each run replaces the CSV.
    with open(EDGES_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "relationship"])
        writer.writerows(edges)

    print(f"Completed. {len(edges)} edges saved to: {EDGES_CSV_PATH}")


if __name__ == "__main__":
    main()
