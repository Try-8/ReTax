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
Year-by-year CSVs are written to the output folder, together with data_dictionary.txt,
which explains every CSV column and how it is calculated.
"""
import argparse
import csv
import os
import sys
from dataclasses import replace

import health
import rules
import statetax
from columns import render_data_dictionary
from model import BUCKET_GROUP, _bucket_rates
from model import STRATEGIES, _ss_persons, load_params, simulate, solve_max_w

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


def fmt_age(months):
    return f"{months // 12}y{months % 12}m"


def _ss_people(p):
    """(person number, birth date, FRA monthly benefit, claim months) for each SS recipient."""
    for n, birth, amt, claim in ((1, p.person1_birth, p.ss_fra_monthly1, p.ss_claim_months1),
                                 (2, p.person2_birth, p.ss_fra_monthly2, p.ss_claim_months2)):
        if birth is not None and amt > 0:
            yield n, birth, amt, claim


def ss_start_notes(p):
    """{year: note} marking each person's first (possibly partial) benefit year."""
    by_year = {}
    for n, birth, _, claim in _ss_people(p):
        y, _m = rules.ss_start_year_month(birth, claim)
        by_year.setdefault(y, []).append(f"person {n}")
    return {y: "Social Security starts: " + " & ".join(v) for y, v in by_year.items()}


def life_event_notes(p):
    """{year: note} for the first death and the switch to single filing."""
    notes = {}
    if p.person2_birth is None:
        return notes
    d1, d2 = p.death1, p.death2
    first_person, first = (1, d1) if (d1.year, d1.month) <= (d2.year, d2.month) else (2, d2)
    if p.last_death_year > first.year:
        notes[first.year] = f"person {first_person} dies {MONTHS[first.month - 1]} {first.year} (survivor benefit begins)"
        if p.filing_status == "mfj" and first.year + 1 < p.last_death_year:
            notes[first.year + 1] = "files SINGLE from here"
        elif p.filing_status == "mfj" and first.year + 1 == p.last_death_year:
            notes[first.year + 1] = "(survivor's final year is not simulated)"
    return notes


def state_move_notes(p):
    """{year: note} for the year of a move to another state."""
    if p.move_date is None or not (p.start_year <= p.move_date.year < p.start_year + p.n_years):
        return {}
    return {p.move_date.year: f"moves {p.state_origin} -> {p.state_dest} ({MONTHS[p.move_date.month - 1]} "
                              f"{p.move_date.year}; state tax prorated by month)"}


def inheritance_notes(p):
    """{year: note} for each parent's death, asset sales, and the inherited IRA's 10-year deadline."""
    notes = {}
    last = p.start_year + p.n_years - 1

    def add(year, text):
        if p.start_year <= year <= last:
            notes[year] = (notes[year] + "; " + text) if year in notes else text

    for ev in p.inheritances:
        if p.death_year(1) < ev.death_date.year:
            continue
        d = ev.death_date
        price = (1 + p.inflation) ** (d.year - p.start_year)
        ira = ev.ira_value * ev.share * price
        add(d.year, f"your {ev.parent} dies {MONTHS[d.month - 1]} {d.year}" +
            (f": inherited IRA {money(ira)} received" if ira else ""))
        for a in ev.assets:
            idx = d.year * 12 + d.month - 1 + a.months_after
            net = a.value * ev.share * (1 + p.inflation) ** (idx // 12 - p.start_year) * (1 - a.cost_pct)
            add(idx // 12, f"{a.label}: {money(net)} net proceeds received")
        if ev.ira_value:
            add(d.year + rules.INHERITED_IRA_YEARS, f"inherited IRA ({ev.parent}) must be EMPTY by Dec 31")
    return notes


def medicare_start_notes(p):
    """{year: note} for the year each person turns 65 and Medicare begins."""
    if not p.health_enabled:
        return {}
    by_year = {}
    for n, birth in ((1, p.person1_birth), (2, p.person2_birth)):
        if birth is None:
            continue
        y = (birth.year * 12 + birth.month - 1 + rules.MEDICARE_AGE_MONTHS) // 12
        if p.start_year < y <= p.death_year(n) and y < p.start_year + p.n_years:
            by_year.setdefault(y, []).append(f"person {n}")
    return {y: "Medicare begins: " + " & ".join(v) for y, v in by_year.items()}


def rmd_start_notes(p):
    """{year: note} for the first year each person must take an RMD."""
    by_year = {}
    for n, birth in ((1, p.person1_birth), (2, p.person2_birth)):
        if birth is None:
            continue
        y = birth.year + rules.rmd_age(birth.year)
        if p.start_year < y <= p.death_year(n) and y < p.start_year + p.n_years:
            by_year.setdefault(y, []).append(f"person {n}")
    return {y: "RMDs begin: " + " & ".join(v) for y, v in by_year.items()}


def show_run(title, res, p, every=5):
    notes = {}
    for source in (ss_start_notes(p), rmd_start_notes(p), medicare_start_notes(p),
                   life_event_notes(p), state_move_notes(p), inheritance_notes(p)):
        for y, text in source.items():
            notes[y] = (notes[y] + "; " + text) if y in notes else text
    print(f"\n--- {title} ---")
    status = "money lasts to horizon" if res.success else f"RUNS OUT in {res.depletion_year}"
    print(f"Status: {status}")
    print(f"Ending balance: {money(res.final_real)} real / {money(res.final_nominal)} nominal")
    print(f"Lifetime federal income tax (nominal): {money(res.lifetime_tax)}")
    print(f"Lifetime state income tax (nominal):   {money(res.lifetime_state_tax)}")
    if p.conv_bracket is not None:
        print(f"Roth conversions (nominal):            {money(res.lifetime_conversions)} moved to Roth")
    if p.asset_classes:
        income = sum(r["inv_income_ordinary"] + r["inv_income_qualified"] for r in res.rows)
        print(f"Dividends and interest taxed in cash/taxable (nominal): {money(income)}")
    if p.inheritances:
        print(f"Inheritances received (nominal):       {money(res.lifetime_inheritance)}")
    if p.health_enabled:
        print(f"Lifetime health premiums (nominal):    {money(res.lifetime_health)}"
              f" (of which Medicare IRMAA surcharges {money(res.lifetime_irmaa)})")
    penalty_banner(res)
    print("(Dollar columns are NOMINAL unless marked real. Compare SocSec to Spend(nom) + Health(nom), "
          "not Spend(real).\n Penalties are flagged in the notes and in the CSV.)")
    print(f"{'Year':>5} {'Age':>3} {'Filing':>6} {'Spend(real)':>11} {'Spend(nom)':>11} {'SocSec(nom)':>11} "
          f"{'InhIRA(nom)':>12} {'RMD(nom)':>9} {'Conv(nom)':>10} {'DivInt(nom)':>11} {'FedTax(nom)':>11} {'StTax(nom)':>10} {'Health(nom)':>11} {'Marg':>4} {'Balance(real)':>13}")
    prev_tier = 0
    for i, r in enumerate(res.rows):
        tags = []
        if r["early_withdrawal_flag"]:
            tags.append("PENALTY")
        if r["irmaa_tier"] != prev_tier:
            tags.append(f"IRMAA tier {r['irmaa_tier']}")
        prev_tier = r["irmaa_tier"]
        if r["year"] in notes:
            tags.append(notes[r["year"]])
        if i % every == 0 or i == len(res.rows) - 1 or tags:
            suffix = ("  <-- " + "; ".join(tags)) if tags else ""
            print(f"{r['year']:>5} {r['age_person1']:>3} {r['filing_status'].upper():>6} {money(r['spend_real']):>11} "
                  f"{money(r['spend_nominal']):>11} {money(r['ss_gross']):>11} "
                  f"{money(r['draw_inherited_ira']):>12} {money(r['rmd_required']):>9} {money(r['roth_conversion']):>10} {money(r['inv_income_ordinary'] + r['inv_income_qualified']):>11} {money(r['tax_federal']):>11} {money(r['tax_state']):>10} "
                  f"{money(r['health_total']):>11} "
                  f"{r['marginal_ordinary_rate']:>4.0%} {money(r['end_total_real']):>13}{suffix}")


def describe_timeline(p):
    print("\nSocial Security (claim age is set in [social_security] of your config):")
    for n, birth, amt, claim in _ss_people(p):
        fra = rules.fra_months(birth.year)
        claim_m = fra if claim is None else claim
        factor = rules.ss_claim_factor(birth, claim)
        y, m = rules.ss_start_year_month(birth, claim)
        first_months = rules.ss_months_paid(birth, y, claim)
        print(f"  Person {n}: FRA {fmt_age(fra)}, claims at {fmt_age(claim_m)} -> "
              f"{factor:.1%} of the FRA benefit = {money(amt * factor * p.ss_multiplier)}/month (today's $)")
        print(f"            first benefit month {MONTHS[m - 1]} {y} "
              f"({first_months} months of benefits in {y}, 12 in later years)")
    if p.ss_multiplier != 1.0:
        print(f"  Benefit multiplier applied: {p.ss_multiplier:.2f}")
    if p.person2_birth is not None:
        if not p.ss_spousal:
            print("  Spousal top-up: switched off in the config.")
        else:
            tops = rules.spousal_topups(_ss_persons(p))
            if not any(t["excess"] > 0 for t in tops):
                print("  Spousal top-up: none (each person's own benefit is at least half of the other's).")
            for n, top in enumerate(tops, start=1):
                if top["excess"] <= 0:
                    continue
                other = 3 - n
                own, other_pia = (p.ss_fra_monthly1, p.ss_fra_monthly2) if n == 1 else (p.ss_fra_monthly2, p.ss_fra_monthly1)
                y, m = divmod(top["start"], 12)
                cut = "" if top["factor"] >= 1.0 else f", reduced {1 - top['factor']:.1%} for starting before their full retirement age"
                print(f"  Spousal top-up: person {n} also gets {money(top['monthly'])}/month (today's $) from "
                      f"{MONTHS[m]} {y}: half of person {other}'s benefit at full retirement age "
                      f"({money(0.5 * other_pia)}) exceeds their own ({money(own)}). It starts when BOTH have "
                      f"claimed{cut}, and ends at the first death.")
    print("\nDeath dates (the plan runs to the last death; default is the "
          f"{p.horizon_age}th birthday):")
    for n, birth, death in ((1, p.person1_birth, p.death1), (2, p.person2_birth, p.death2)):
        if birth is not None:
            given = (p.person1_death if n == 1 else p.person2_death) is not None
            age = (death.year * 12 + death.month - birth.year * 12 - birth.month) // 12
            print(f"  Person {n}: {MONTHS[death.month - 1]} {death.year} (age {age})"
                  f"{'' if given else '  [default]'}")
    if p.person2_birth is not None:
        fd = p.first_death_year
        if p.filing_status == "mfj" and fd + 1 < p.last_death_year:
            print(f"  Filing: joint through {fd} (year of first death), single from {fd + 1}.")
        else:
            print(f"  Filing: joint throughout (the first death year is the last simulated year or later).")
        d1, d2 = p.death1, p.death2
        idx1, idx2 = d1.year * 12 + d1.month, d2.year * 12 + d2.month
        dec_n, dec_birth, dec_claim, dec_amt, dec_death, sur_birth = (
            (1, p.person1_birth, p.ss_claim_months1, p.ss_fra_monthly1, d1, p.person2_birth) if idx1 <= idx2
            else (2, p.person2_birth, p.ss_claim_months2, p.ss_fra_monthly2, d2, p.person1_birth))
        d_idx = rules.month_index(dec_death.year, dec_death.month)
        basis = rules.survivor_basis_factor(dec_birth, dec_claim, d_idx)
        s_start, s_redn = rules.survivor_start_and_reduction(sur_birth, d_idx)
        sy, sm = divmod(s_start, 12)
        print(f"  Survivor benefit (after person {dec_n} dies): {basis:.1%} of their FRA benefit "
              f"= {money(dec_amt * basis * s_redn * p.ss_multiplier)}/month (today's $), "
              f"from {MONTHS[sm]} {sy}"
              f"{'' if s_redn == 1.0 else f' (reduced {1 - s_redn:.1%} for taking it before FRA)'}; "
              f"the household receives the larger of this and the survivor's own benefit.")
    print("Roth conversions:")
    if p.conv_bracket is None:
        print("  none (add a [roth_conversion] block to fill a tax bracket each year in a window)")
    else:
        print(f"  {p.conv_start}-{p.conv_end}: each year convert your own traditional IRAs to Roth up to the top of "
              f"the {p.conv_bracket:.0%} federal bracket (counting Social Security the conversion makes taxable); "
              f"the tax is paid from cash/taxable first. Inherited IRAs are not converted.")
    print("Asset classes and account mixes:")
    if not p.asset_classes:
        print(f"  none defined: every account earns {p.nominal_return:.2%} and nothing is taxed until sold "
              f"(add [[asset_class]] and [allocation] to model returns by class and tax dividends/interest yearly)")
    else:
        for c in p.asset_classes:
            print(f"  {c.name}: total return {c.total_return:.2%}, income yield {c.income_yield:.2%} "
                  f"({c.qualified_share:.0%} qualified)")
        rates = _bucket_rates(p)
        for group, mix in p.allocations:
            bucket = next(b for b, g in BUCKET_GROUP.items() if g == group)
            total, ordinary, qualified = rates[bucket]
            parts = ", ".join(f"{w:.0%} {name}" for name, w in mix if w > 0)
            print(f"  {group:<12} {parts:<36} -> return {total:.2%}, income {ordinary + qualified:.2%}")
        print("  (dividends and interest in the cash and taxable accounts are taxed every year; "
              "traditional, Roth and inherited accounts defer or skip tax)")
    print("Inheritances (received by person 1; amounts in today's dollars):")
    if not p.inheritances:
        print("  none modeled (add [inheritance.father] / [inheritance.mother] blocks to your config)")
    for ev in p.inheritances:
        d = ev.death_date
        age = f", age {d.year - ev.birth_date.year}" if ev.birth_date else ""
        print(f"  Your {ev.parent} dies {MONTHS[d.month - 1]} {d.year}{age}; your share {ev.share:.0%}.")
        if ev.ira_value:
            factor = rules.single_life_factor(d.year + 1 - p.person1_birth.year)
            print(f"    Traditional IRA {money(ev.ira_value)}: received that year. You owe {'his' if ev.parent == 'father' else 'her'} RMD for the year of "
                  f"death, then annual RMDs from {d.year + 1} (Single Life factor {factor} at your age "
                  f"{d.year + 1 - p.person1_birth.year}, minus 1 each year), and the account must be empty by "
                  f"Dec 31, {d.year + rules.INHERITED_IRA_YEARS}. Drawn '{p.inherited_ira_policy}'.")
        for a in ev.assets:
            print(f"    {a.label}: {money(a.value)}, sold {a.months_after} months after death, "
                  f"{a.cost_pct:.0%} selling costs; net proceeds go to your taxable account (stepped-up basis).")
    print("Health insurance (paid on top of living spending W):")
    if not p.health_enabled:
        print("  not modeled (add a [healthcare] block to your config to include it)")
    else:
        for n, birth in ((1, p.person1_birth), (2, p.person2_birth)):
            if birth is not None:
                idx = birth.year * 12 + birth.month - 1 + rules.MEDICARE_AGE_MONTHS
                print(f"  Person {n}: marketplace {money(p.health_marketplace_monthly)}/month (today's $) "
                      f"until Medicare starts {MONTHS[idx % 12]} {idx // 12}")
        other = (f" + {money(p.health_medicare_other_monthly)}/month other premiums"
                 if p.health_medicare_other_monthly else "")
        print(f"  From 65: Medicare Part B (${health.PART_B_STANDARD:,.2f}/month in 2026 dollars){other} plus "
              f"IRMAA, set by AGI from two years earlier against the thresholds for that year's filing status.")
        if p.health_premium_growth:
            print(f"  Premiums grow {p.health_premium_growth:.1%} a year above inflation.")
    print("State income tax:")
    if p.state_origin is None:
        print("  none modeled (add a [state] block to your config to include one)")
    else:
        name = statetax.STATES[p.state_origin]["name"]
        r0 = statetax.flat_rate(p.state_origin, p.start_year)
        rl = statetax.flat_rate(p.state_origin, p.start_year + p.n_years)
        rates = f"{r0:.2%} in {p.start_year}" + (f", {rl:.2%} later" if rl != r0 else "")
        local = f", plus {p.local_tax_rate:.2%} local (county) tax" if p.local_tax_rate else ""
        print(f"  Living in {name} ({p.state_origin}): flat rate {rates}{local}. Social Security is "
              f"not taxed by Indiana; IRA withdrawals and capital gains are." if p.state_origin == "IN"
              else f"  Living in {name} ({p.state_origin}): flat rate {rates}{local}.")
        if p.move_date is not None:
            dn = statetax.STATES[p.state_dest]["name"]
            print(f"  Moves to {dn} ({p.state_dest}) in {MONTHS[p.move_date.month - 1]} {p.move_date.year}; "
                  f"that year's state tax is prorated by month; {dn} rate afterwards: "
                  f"{statetax.flat_rate(p.state_dest, p.move_date.year):.2%}.")
    print("Required minimum distributions (IRS Uniform Lifetime Table; Roth accounts have none):")
    for n, birth in ((1, p.person1_birth), (2, p.person2_birth)):
        if birth is None:
            continue
        age, y = rules.rmd_age(birth.year), birth.year + rules.rmd_age(birth.year)
        if y > p.death_year(n):
            when = f"age {age} ({y}) -- after the death date, so none"
        elif y <= p.start_year:
            when = f"age {age}: already required at the start of the model"
        else:
            when = f"start at age {age} ({y})"
        print(f"  Person {n}: {when}")
    if p.person2_birth is not None:
        for n, birth, other in ((1, p.person1_birth, p.person2_birth), (2, p.person2_birth, p.person1_birth)):
            if (other.year * 12 + other.month) - (birth.year * 12 + birth.month) > 120:
                print(f"  Note: person {n}'s spouse is more than 10 years younger. If the spouse is the sole "
                      f"beneficiary the IRS Joint Life table (smaller RMDs) would apply; the model uses the "
                      f"Uniform table, so person {n}'s RMDs are slightly overstated.")
    print("Retirement accounts penalty-free starting (59.5 rule):")
    for n, birth in ((1, p.person1_birth), (2, p.person2_birth)):
        if birth is not None:
            print(f"  Person {n}: {p.unlock_year(n)}")


def compare_strategies(p):
    t = p.target_monthly_real
    print("\n=== Strategy comparison (flat spending; same inputs, different withdrawal plan) ===")
    print(f"{'Strategy':<26} {'W flat/mo':>10} | at {money(t)}/mo target: "
          f"{'fed+state tax':>13} {'penalty':>9} {'end balance (real)':>19}")
    for name in STRATEGIES:
        q = replace(p, strategy=name)
        w = solve_max_w(q)
        r = simulate(q, t)
        mark = "*" if name == p.strategy else " "
        end = money(r.final_real) if r.success else f"runs out {r.depletion_year}"
        print(f"{mark}{name:<25} {money(w):>10} | {'':>22}{money(r.lifetime_tax + r.lifetime_state_tax):>13} "
              f"{money(r.lifetime_penalty):>9} {end:>19}")
    print("(* = strategy selected in your config. Differences of 1-3% are within the noise "
          "of the assumptions.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.local.toml")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"Config '{args.config}' not found. Copy config.template.toml to "
                 f"config.local.toml and fill in your numbers.")
    try:
        p = load_params(args.config)
    except (ValueError, KeyError) as e:
        sys.exit(f"Config problem: {e}")
    os.makedirs(args.out, exist_ok=True)

    print("=== Retirement model v9 ===")
    print(f"Years simulated: {p.start_year}-{p.start_year + p.n_years - 1} ({p.n_years} years), "
          f"starting balance {money(p.total_start_balance)}")
    ret = ("Returns by asset class (see below)" if p.asset_classes
           else f"Return {p.nominal_return:.2%} nominal")
    print(f"{ret}, inflation {p.inflation:.2%}, "
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

    if p.inheritances:
        w_noinh = solve_max_w(replace(p, inheritances=()), use_curve=False)
        print(f"W, flat, without inheritances: {money(w_noinh)}/month "
              f"(the inheritances add {money(w_flat - w_noinh)}/month of W)")

    compare_strategies(p)

    # Metrics 2 and 3: balance paths at the target spending level
    t = p.target_monthly_real
    print(f"\nTarget spending for balance paths: {money(t)}/month real")
    flat = simulate(p, t, use_curve=False)
    curve = simulate(p, t, use_curve=True)
    show_run("Metric 2: flat spending", flat, p)
    show_run("Metric 3: spending curve", curve, p)

    write_csv(os.path.join(args.out, "flat_path.csv"), flat.rows)
    write_csv(os.path.join(args.out, "curve_path.csv"), curve.rows)
    dict_path = os.path.join(args.out, "data_dictionary.txt")
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write(render_data_dictionary(p))
    print(f"\nYear-by-year CSVs and data_dictionary.txt (explains every column) written to '{args.out}/'")


if __name__ == "__main__":
    main()
