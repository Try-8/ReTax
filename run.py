"""
run.py -- Run the model against your local config file.

Usage:
    python run.py                          # reads config.local.toml
    python run.py --config other.toml
    python run.py --out outputs

Reports:
  Metric 1  W              max constant monthly real after-tax spending
  Metric 2  Flat run       balance path at target_monthly_real, flat spending
  Metric 3  Curve run      balance path at target_monthly_real x spending curve
  Plus      Strategy comparison (W and taxes for each built-in withdrawal plan)
Early-withdrawal penalties (10%) are flagged wherever they occur.
Year-by-year CSVs are written to the output folder.
"""
import argparse
import csv
import os
import sys
from dataclasses import replace

import rules
from model import STRATEGIES, load_params, simulate, solve_max_w

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def money(x):
    return f"${x:,.0f}"


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def penalty_banner(res):
    if res.lifetime_penalty > 0.005:
        yrs = res.penalty_years
        span = f"{yrs[0]}" if len(yrs) == 1 else f"{yrs[0]}-{yrs[-1]} ({len(yrs)} years)"
        print(f"  !! EARLY-WITHDRAWAL PENALTY: {money(res.lifetime_penalty)} total, "
              f"incurred in {span}. Retirement accounts were drawn before 59.5 "
              f"because no other money was available.")
    else:
        print("  No early-withdrawal penalty.")


def show_run(title, res, every=5):
    print(f"\n--- {title} ---")
    status = "money lasts to horizon" if res.success else f"RUNS OUT in {res.depletion_year}"
    print(f"Status: {status}")
    print(f"Ending balance: {money(res.final_real)} real / {money(res.final_nominal)} nominal")
    print(f"Lifetime federal income tax (nominal): {money(res.lifetime_tax)}")
    penalty_banner(res)
    print(f"{'Year':>6} {'Age':>4} {'Spend(real)':>12} {'SocSec':>9} {'SSTaxed':>9} "
          f"{'Tax':>9} {'Penalty':>9} {'Marg':>5} {'Balance(real)':>14}")
    for i, r in enumerate(res.rows):
        if i % every == 0 or i == len(res.rows) - 1 or r["early_withdrawal_flag"]:
            flag = " <-- PENALTY" if r["early_withdrawal_flag"] else ""
            print(f"{r['year']:>6} {r['age_person1']:>4} {money(r['spend_real']):>12} "
                  f"{money(r['ss_gross']):>9} {money(r['ss_taxable']):>9} "
                  f"{money(r['tax_federal']):>9} {money(r['penalty_early_withdrawal']):>9} "
                  f"{r['marginal_ordinary_rate']:>5.0%} {money(r['end_total_real']):>14}{flag}")


def describe_timeline(p):
    print("\nSocial Security (starts at Full Retirement Age):")
    for n, birth, amt in ((1, p.person1_birth, p.ss_fra_monthly1),
                          (2, p.person2_birth, p.ss_fra_monthly2)):
        if birth is None or amt <= 0:
            continue
        y, m = rules.ss_start_year_month(birth)
        print(f"  Person {n}: {money(amt)}/month (today's $) starting {MONTHS[m - 1]} {y}")
    if p.ss_multiplier != 1.0:
        print(f"  Benefit multiplier applied: {p.ss_multiplier:.2f}")
    print("Retirement accounts penalty-free starting (59.5 rule):")
    for n, birth in ((1, p.person1_birth), (2, p.person2_birth)):
        if birth is not None:
            print(f"  Person {n}: {p.unlock_year(n)}")


def compare_strategies(p):
    t = p.target_monthly_real
    print("\n=== Strategy comparison (flat spending; same inputs, different withdrawal plan) ===")
    print(f"{'Strategy':<26} {'W flat/mo':>10} | at {money(t)}/mo target: "
          f"{'lifetime tax':>13} {'penalty':>9} {'end balance (real)':>19}")
    for name in STRATEGIES:
        q = replace(p, strategy=name)
        w = solve_max_w(q)
        r = simulate(q, t)
        mark = "*" if name == p.strategy else " "
        end = money(r.final_real) if r.success else f"runs out {r.depletion_year}"
        print(f"{mark}{name:<25} {money(w):>10} | {'':>22}{money(r.lifetime_tax):>13} "
              f"{money(r.lifetime_penalty):>9} {end:>19}")
    print("(* = strategy selected in your config. Differences of 1-3% are within the noise "
          "of the assumptions.)")


def main():
    # read the command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.local.toml")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    # give error if config argument file is not found
    if not os.path.exists(args.config):
        sys.exit(f"Config '{args.config}' not found. Copy config.template.toml to "
                 f"config.local.toml and fill in your numbers.")

    #try loading the configuration file data
    try:
        p = load_params(args.config)
    except (ValueError, KeyError) as e:
        sys.exit(f"Config problem: {e}")
    os.makedirs(args.out, exist_ok=True)

    print("=== Retirement model v1 ===")
    print(f"Years simulated: {p.start_year}-{p.start_year + p.n_years - 1} ({p.n_years} years), "
          f"starting balance {money(p.total_start_balance)}")
    print(f"Return {p.nominal_return:.2%} nominal, inflation {p.inflation:.2%}, "
          f"filing {p.filing_status}, taxes {'on' if p.tax_enabled else 'OFF'}")
    print(f"Withdrawal strategy: {p.strategy}")
    describe_timeline(p)

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

    compare_strategies(p)

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
