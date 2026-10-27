"""Step 1: ask the LLM to extract variables, constants and equations from each
Fortran file and write everything to MATH_TXT_PATH. Connection settings live in
api_config.py."""

import os

from api_config import FORTRAN_FOLDER, MATH_TXT_PATH, chat_with_llm


def read_fortran_file(file_path):
    """Read a source file, falling back to latin-1 for non-UTF-8 files."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def build_prompt(file_name, fortran_code):
    # The section headers below are parsed by symbolic_math_to_csv.py - keep them in sync.
    return (
        f"From the Fortran file `{file_name}`, extract the following in a clear and structured way:\n\n"
        "1. Global Variables: variables defined at module or program level\n"
        "2. Local Variables: variables defined inside subroutines or functions\n"
        "3. Constants: named constants used in equations\n"
        "4. Equations: write the full mathematical expression in human-readable format\n\n"
        "Use this exact output format:\n\n"
        "### Global Variables:\n"
        "VariableName (symbol)\n"
        "...\n\n"
        "### Local Variables:\n"
        "VariableName (symbol)\n"
        "...\n\n"
        "### Constants:\n"
        "ConstantName (symbol)\n"
        "...\n\n"
        "### Equations:\n"
        "EquationName (equation)\n"
        "...\n\n"
        "Here is the Fortran code:\n\n"
        f"{fortran_code}"
    )


def main():
    if not os.path.isdir(FORTRAN_FOLDER):
        raise SystemExit(f"Fortran folder not found: {FORTRAN_FOLDER!r} (set FORTRAN_FOLDER in api_config.py)")

    # Collect all .f90/.F90 files, including subdirectories (case-insensitive).
    fortran_files = [
        os.path.join(root, name)
        for root, _, names in os.walk(FORTRAN_FOLDER)
        for name in names
        if name.lower().endswith(".f90")
    ]
    print(f"Total Fortran files found: {len(fortran_files)}")

    processed = 0
    # The output directory must already exist; each run replaces the output file.
    with open(MATH_TXT_PATH, "w", encoding="utf-8") as txt_file:
        txt_file.write("===== Extracted Global/Local Variables, Constants, and Equations =====\n\n")

        for index, file_path in enumerate(fortran_files, start=1):
            # Only the basename is retained, so identical names in different
            # source directories share the same file identity downstream.
            file_name = os.path.basename(file_path)
            print(f"Processing {index}/{len(fortran_files)}: {file_name}")

            try:
                response = chat_with_llm(build_prompt(file_name, read_fortran_file(file_path)))
            except RuntimeError as e:
                print(f" Skipped {file_name}: {e}")
                continue

            if not response.strip():
                print(f" No data extracted for: {file_name}")
                continue

            txt_file.write(f"===== File: {file_name} =====\n\n")
            txt_file.write(response)
            txt_file.write("\n" + "=" * 80 + "\n\n")
            txt_file.flush()  # keep partial results if the run is interrupted
            processed += 1

    print(f"\nExtracted {processed} of {len(fortran_files)} files.")
    print(f"TXT file created: {MATH_TXT_PATH}")


if __name__ == "__main__":
    main()
