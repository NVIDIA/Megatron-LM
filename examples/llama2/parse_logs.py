import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog="Process Log")
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()
    lines = lines[1:-1]
    lines = [float(a) for a in lines]
    mean = np.mean(np.array(lines))
    print(mean)'