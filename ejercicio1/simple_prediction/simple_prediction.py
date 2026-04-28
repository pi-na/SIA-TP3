"""Aplica reglas duras de detección de fraude al fraud_dataset.

Reglas determinísticas derivadas del análisis exploratorio (ver
ejercicio1/analisis_desarrollado.md):

    amount_usd > 500
    quantity_purchased >= 10
    items_viewed_before_purchase >= 15

Cada regla, individualmente, separa perfectamente fraude de no-fraude
(precision = 100%) en el dataset original. Se aplican como OR.

La regla candidata `session_duration_seconds > 500` se descartó porque
captura 1 TP y 350 FP (rompe la precision perfecta).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


RULES = {
    "amount_usd > 500": lambda df: df["amount_usd"] > 500,
    "quantity_purchased >= 10": lambda df: df["quantity_purchased"] >= 10,
    "items_viewed_before_purchase >= 15": lambda df: df["items_viewed_before_purchase"] >= 15,
}


def predict(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for rule_fn in RULES.values():
        mask = mask | rule_fn(df)
    return mask.astype(int)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def per_rule_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """TP/FP por cada regla individual. Útil para debugging."""
    if "flagged_fraud" not in df.columns:
        return pd.DataFrame()
    rows = []
    for name, fn in RULES.items():
        mask = fn(df)
        tp = int((mask & (df["flagged_fraud"] == 1)).sum())
        fp = int((mask & (df["flagged_fraud"] == 0)).sum())
        rows.append({"rule": name, "matches": int(mask.sum()), "tp": tp, "fp": fp})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Ruta al CSV de entrada.")
    parser.add_argument("--output", default=None,
                        help="Ruta del CSV de salida (default: <csv>_with_prediction.csv).")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"ERROR: no existe {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["simple_prediction"] = predict(df)

    output_path = Path(args.output) if args.output else csv_path.with_name(
        csv_path.stem + "_with_prediction.csv"
    )
    df.to_csv(output_path, index=False)
    print(f"CSV guardado en: {output_path}")
    print(f"Filas predichas como fraude: {int(df['simple_prediction'].sum())} / {len(df)}")

    if "flagged_fraud" in df.columns:
        m = compute_metrics(df["flagged_fraud"], df["simple_prediction"])
        print()
        print("=== Métricas vs flagged_fraud ===")
        print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}  total={m['total']}")
        print(f"  accuracy  = {m['accuracy']:.4f}")
        print(f"  precision = {m['precision']:.4f}")
        print(f"  recall    = {m['recall']:.4f}")
        print()
        print("=== Aporte por regla individual ===")
        print(per_rule_breakdown(df).to_string(index=False))


if __name__ == "__main__":
    main()
