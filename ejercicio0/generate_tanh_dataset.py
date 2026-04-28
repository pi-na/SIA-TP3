import argparse
import numpy as np
import pandas as pd


def generate_dataset(n_points=50, seed=None):
    """Generate y = tanh(x) dataset (no noise — exact function for validation)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, n_points)
    y = np.tanh(x)
    return pd.DataFrame({"x": x, "y": y})


def main():
    parser = argparse.ArgumentParser(description="Generate y=tanh(x) dataset")
    parser.add_argument("--output", default="tanh_dataset.csv", help="Output CSV path")
    parser.add_argument("--n_points", type=int, default=50, help="Number of data points")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    df = generate_dataset(n_points=args.n_points, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Dataset saved to {args.output} ({len(df)} points)")


if __name__ == "__main__":
    main()
