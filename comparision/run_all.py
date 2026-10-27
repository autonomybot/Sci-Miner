"""Run the whole comparison:  python run_all.py [--quick]"""
import subprocess, sys, os

here = os.path.dirname(os.path.abspath(__file__))
quick = ["--quick"] if "--quick" in sys.argv else []
steps = [["path_recovery_comparison.py", "--rebuild"],
         ["regression_comparison.py", "--cases", "methane"] + quick,
         ["regression_comparison.py", "--cases", "lake"] + quick,
         ["make_report.py"],
         ["plot_comparison_figures.py"]]
for s in steps:
    print(">>", " ".join(s), flush=True)
    subprocess.run([sys.executable] + s, cwd=here, check=True)
