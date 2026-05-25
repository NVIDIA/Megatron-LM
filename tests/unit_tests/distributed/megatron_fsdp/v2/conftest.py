import os
import sys


def _redirect_nonzero_ranks():
    rank = int(os.getenv("RANK", "0"))
    if rank != 0:
        os.makedirs("pytest_ranks", exist_ok=True)
        f = open(f"pytest_ranks/rank{rank}.out", "w", buffering=1)
        sys.stdout = f
        sys.stderr = f


_redirect_nonzero_ranks()
