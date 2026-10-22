"""
run.py -- Run the v0 model against your local config file.

Usage:
    python run.py                          # reads config.local.toml
    python run.py --config other.toml
    python run.py --out outputs

Reports three metrics:
  1. W            max constant monthly real after-tax spending (money lasts to horizon)
  2. Flat run     balance path at target_monthly_real, flat real spending
  3. Curve run    balance path at target_monthly_real x spending curve
Also writes year-by-year CSVs to the output folder.
"""
import argparse
import csv
import os
import sys

from model import load_params, simulate, solve_max_w


def money(x):
    return f"${x:,.0f}"


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def show_run(title, res, every=5):
    print(f"\n--- {title} ---")
    status = "money lasts to horizon" if res.success else f"RUNS OUT in {res.depletion_year}"
    print(f"Status: {status}")
    print(f"Ending balance: {money(res.final_real)} real / {money(res.final_nominal)} nominal")
    print(f"Lifetime federal tax (nominal): {money(res.lifetime_tax)}")
    print(f"{'Year':>6} {'Age':>4} {'Spend(real)':>12} {'Tax(nom)':>10} {'MargRate':>9} {'Balance(real)':>14}")
    for i, r in enumerate(res.rows):
        if i % every == 0 or i == len(res.rows) - 1:
            print(f"{r['year']:>6} {r['age_person1']:>4} {money(r['spend_real']):>12} "
                  f"{money(r['tax_federal']):>10} {r['marginal_ordinary_rate']:>9.0%} "
                  f"{money(r['end_total_real']):>14}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.local.toml")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"Config '{args.config}' not found. Copy config.template.toml to "
                 f"config.local.toml and fill in your numbers.")
    p = load_params(args.config)
    os.makedirs(args.out, exist_ok=True)

    print("=== Retirement model v0 ===")
    print(f"Years simulated: {p.start_year}-{p.start_year + p.n_years - 1} ({p.n_years} years), "
          f"starting balance {money(p.total_start_balance)}")
    print(f"Return {p.nominal_return:.2%} nominal, inflation {p.inflation:.2%}, "
          f"filing {p.filing_status}, taxes {'on' if p.tax_enabled else 'OFF'}")
    print(f"Withdrawal order: {' -> '.join(p.withdrawal_order)}")

    # Metric 1: W
    w_flat = solve_max_w(p, use_curve=False)
    w_curve = solve_max_w(p, use_curve=True)
    print("\n=== Metric 1: W (monthly, real, after-tax) ===")
    print(f"W, flat spending:            {money(w_flat)}/month")
    print(f"W, with spending curve:      {money(w_curve)}/month at curve multiplier 1.0")
    if p.purchases:
        w_no = solve_max_w(p, use_curve=False, include_purchases=False)
        print(f"W, flat, without purchases:  {money(w_no)}/month "
              f"(purchases cost {money(w_no - w_flat)}/month of W)")
        for (y, a, label) in p.purchases:
            print(f"   purchase: {y} {money(a)} real  {label}")

    # Metrics 2 and 3: balance paths at the target spending level
    t = p.target_monthly_real
    print(f"\nTarget spending for balance paths: {money(t)}/month real")
    flat = simulate(p, t, use_curve=False)
    curve = simulate(p, t, use_curve=True)
    show_run("Metric 2: flat spending", flat)
    show_run("Metric 3: spending curve", curve)

    write_csv(os.path.join(args.out, "flat_path.csv"), flat.rows)
    write_csv(os.path.join(args.out, "curve_path.csv"), curve.rows)
    print(f"\nYear-by-year CSVs written to '{args.out}/'")


if __name__ == "__main__":
    main()
