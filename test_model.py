"""
Tests for the retirement model.

Run from this folder:
    python -m unittest                      # fast loop (slow tests are skipped)
    SLOW_TESTS=1 python -m unittest -v      # everything, before you trust a change
    python -m unittest --durations 10       # see which tests are slow (Python 3.12+)
    UPDATE_GOLDEN=1 python -m unittest      # re-record the golden regression file

When to update the golden file: ONLY after a change that you intend to shift
the numbers (e.g. you corrected a tax table). Review the diff first -- an
unexpected golden failure is the test doing its job.

Notes on tolerances
  * Year rows returned by simulate() are rounded to cents, so row-level
    accounting identities hold to ~0.03, not 1e-6. The test
    `test_year_level_identity_to_float_precision` checks the identity at full
    float precision by calling the engine's per-year solver directly.
"""
import dataclasses
import json
import os
import random
import unittest
from dataclasses import replace
from datetime import date

import model
import rules
import taxes
from model import Params, simulate, solve_max_w

slow = unittest.skipUnless(os.getenv("SLOW_TESTS"), "slow test: set SLOW_TESTS=1 to run")
ROW_CENT_TOL = 0.03   # rows are rounded to cents; up to ~5 rounded terms are summed


def make_params(**overrides):
    """Build Params from an EXPLICIT field list. Every field is spelled out here
    on purpose: if a new field is added to Params, this helper fails loudly until
    it is added below, so tests never silently inherit a new default."""
    fields = {
        "start_year": 2027,
        "person1_birth": date(1965, 1, 1),
        "person2_birth": None,
        "horizon_age": 95,
        "horizon_person": "person1",
        "filing_status": "mfj",
        "nominal_return": 0.05,
        "inflation": 0.025,
        "tax_enabled": True,
        "cash": 0.0,
        "taxable_value": 0.0,
        "taxable_basis": 0.0,
        "trad1": 0.0,
        "trad2": 0.0,
        "roth1": 0.0,
        "roth2": 0.0,
        "target_monthly_real": 0.0,
        "curve": ((0.0, 1.0),),
        "purchases": (),
        "strategy": "fixed_order",
        "fixed_order": ("cash", "taxable", "trad", "roth"),
        "early_access_order": ("roth", "trad"),
        "ss_fra_monthly1": 0.0,
        "ss_fra_monthly2": 0.0,
        "ss_multiplier": 1.0,
    }
    real = {f.name for f in dataclasses.fields(Params)}
    missing, unknown = real - set(fields), set(fields) - real
    assert not missing, f"make_params is missing Params fields: {sorted(missing)}"
    assert not unknown, f"make_params lists fields Params no longer has: {sorted(unknown)}"
    bad = set(overrides) - real
    assert not bad, f"unknown override(s): {sorted(bad)}"
    fields.update(overrides)
    return Params(**fields)


# ===========================================================================
# Tax functions
# ===========================================================================
class TaxTests(unittest.TestCase):
    def test_ordinary_bracket_math(self):
        tax = taxes.federal_tax(32_200 + 100_000, 0, "mfj")
        self.assertAlmostEqual(tax, 0.10 * 24_800 + 0.12 * (100_000 - 24_800), places=4)

    def test_zero_tax_under_standard_deduction(self):
        self.assertEqual(taxes.federal_tax(30_000, 0, "mfj"), 0.0)

    def test_gains_in_zero_bracket(self):
        self.assertEqual(taxes.federal_tax(0, 100_000, "mfj"), 0.0)

    def test_gains_stack_on_ordinary(self):
        ordinary = 32_200 + 98_900
        self.assertAlmostEqual(
            taxes.federal_tax(ordinary, 10_000, "mfj"),
            taxes.federal_tax(ordinary, 0, "mfj") + 1_500.0, places=4)

    def test_indexing_scales_thresholds(self):
        self.assertAlmostEqual(taxes.federal_tax(2 * 32_200 + 2 * 50_000, 0, "mfj", scale=2.0),
                               2 * taxes.federal_tax(32_200 + 50_000, 0, "mfj"), places=4)


class SocialSecurityTaxTests(unittest.TestCase):
    def test_well_below_first_threshold(self):
        # PI = 10,000 + 10,000 = 20,000 < 32,000
        t, pi = taxes.taxable_social_security(20_000, 10_000, "mfj")
        self.assertEqual(t, 0.0)
        self.assertAlmostEqual(pi, 20_000.0)

    def test_at_first_threshold(self):
        # PI = 20,000 + 12,000 = 32,000: exactly at T1, nothing taxable
        self.assertEqual(taxes.taxable_social_security(24_000, 20_000, "mfj")[0], 0.0)

    def test_fifty_percent_tier(self):
        # PI = 25,000 + 12,000 = 37,000 -> 50% of the 5,000 above 32,000
        t, pi = taxes.taxable_social_security(24_000, 25_000, "mfj")
        self.assertAlmostEqual(t, 2_500.0)
        self.assertAlmostEqual(pi, 37_000.0)

    def test_eighty_five_percent_tier(self):
        # PI = 30,000 + 18,000 = 48,000 -> 0.85*4,000 + 6,000 = 9,400
        self.assertAlmostEqual(taxes.taxable_social_security(36_000, 30_000, "mfj")[0], 9_400.0)

    def test_cap_at_85_percent(self):
        self.assertAlmostEqual(taxes.taxable_social_security(30_000, 100_000, "mfj")[0], 25_500.0)

    def test_single_thresholds(self):
        # PI = 30,000 + 10,000 = 40,000 single (T1 25k, T2 34k):
        # 0.85*(40,000-34,000) + min(0.5*20,000, 0.5*9,000) = 5,100 + 4,500
        self.assertAlmostEqual(taxes.taxable_social_security(20_000, 30_000, "single")[0],
                               0.85 * 6_000 + 4_500.0)

    def test_thresholds_not_indexed_by_scale(self):
        a = taxes.year_tax(30_000, 0, 36_000, "mfj", scale=1.0)
        b = taxes.year_tax(30_000, 0, 36_000, "mfj", scale=2.0)
        self.assertAlmostEqual(a.taxable_ss, b.taxable_ss)


class RulesTests(unittest.TestCase):
    def test_fra_table(self):
        self.assertEqual(rules.fra_months(1955), 66 * 12 + 2)
        self.assertEqual(rules.fra_months(1959), 66 * 12 + 10)
        self.assertEqual(rules.fra_months(1960), 67 * 12)
        self.assertEqual(rules.fra_months(1950), 66 * 12)

    def test_months_paid(self):
        b = date(1965, 6, 15)            # FRA 67 -> June 2032
        self.assertEqual(rules.ss_start_year_month(b), (2032, 6))
        self.assertEqual(rules.ss_months_paid(b, 2031), 0)
        self.assertEqual(rules.ss_months_paid(b, 2032), 7)   # Jun..Dec
        self.assertEqual(rules.ss_months_paid(b, 2033), 12)

    def test_early_unlock_year(self):
        # 59.5 on/before June 30 -> unlocked that calendar year; later -> next year
        self.assertEqual(rules.early_unlock_year(date(1965, 12, 1)), 2025)  # 59.5 = Jun 2025
        self.assertEqual(rules.early_unlock_year(date(1966, 1, 1)), 2026)   # 59.5 = Jul 2025
        self.assertEqual(rules.early_unlock_year(date(1966, 2, 1)), 2026)   # 59.5 = Aug 2025
        self.assertEqual(rules.early_unlock_year(date(1965, 6, 30)), 2025)  # 59.5 = Dec 2024 -> next year


# ===========================================================================
# Params: validation, horizon, ordering
# ===========================================================================
class ParamsTests(unittest.TestCase):
    def test_make_params_default_builds(self):
        self.assertEqual(make_params().n_years, 1965 + 95 - 2027)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            make_params(strategy="nope")

    def test_bad_fixed_order_category_rejected(self):
        with self.assertRaises(ValueError):
            make_params(fixed_order=("cash", "bogus"))

    def test_bad_early_access_order_category_rejected(self):
        with self.assertRaises(ValueError):
            make_params(early_access_order=("cash",))      # only trad/roth allowed

    def test_person2_money_without_person2_birth_rejected(self):
        for field in ("trad2", "roth2", "ss_fra_monthly2"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    make_params(**{field: 1.0}, person2_birth=None)

    def test_horizon_person_person1(self):
        p = make_params(person2_birth=date(1970, 1, 1), horizon_person="person1")
        self.assertEqual(p.horizon_birth_year, 1965)

    def test_horizon_person_person2(self):
        p = make_params(person2_birth=date(1970, 1, 1), horizon_person="person2")
        self.assertEqual(p.horizon_birth_year, 1970)
        self.assertEqual(p.n_years, 1970 + 95 - 2027)

    def test_horizon_person_younger_picks_later_birth_year(self):
        p = make_params(person2_birth=date(1970, 1, 1), horizon_person="younger")
        self.assertEqual(p.horizon_birth_year, 1970)
        q = make_params(person2_birth=date(1960, 1, 1), horizon_person="younger")
        self.assertEqual(q.horizon_birth_year, 1965)        # person 1 is the younger one

    def test_horizon_younger_with_no_person2(self):
        self.assertEqual(make_params(horizon_person="younger").horizon_birth_year, 1965)

    def test_older_person_drawn_first(self):
        older2 = make_params(person2_birth=date(1960, 1, 1))
        self.assertEqual(older2.trad_order, ("trad2", "trad1"))
        self.assertEqual(older2.roth_order, ("roth2", "roth1"))
        older1 = make_params(person2_birth=date(1970, 1, 1))
        self.assertEqual(older1.trad_order, ("trad1", "trad2"))


class CurveTests(unittest.TestCase):
    def test_curve_multiplier_interpolation(self):
        c = ((0, 1.0), (100, 0.5))
        self.assertAlmostEqual(model.curve_multiplier(c, 50), 0.75)
        self.assertAlmostEqual(model.curve_multiplier(c, -10), 1.0)    # flat below range
        self.assertAlmostEqual(model.curve_multiplier(c, 200), 0.5)    # flat above range

    def test_curve_points_may_be_unsorted(self):
        self.assertAlmostEqual(model.curve_multiplier(((100, 0.5), (0, 1.0)), 50), 0.75)

    def test_single_point_curve_is_flat(self):
        self.assertAlmostEqual(model.curve_multiplier(((0.0, 1.0),), 80), 1.0)

    def test_curve_reduces_late_spending_so_w_scale_rises(self):
        p = make_params(roth1=1_000_000.0, strategy="fixed_order", fixed_order=("roth",),
                        curve=((0, 1.0), (95, 0.7)))
        self.assertGreater(solve_max_w(p, use_curve=True), solve_max_w(p, use_curve=False))

    def test_use_curve_changes_spending_rows(self):
        p = make_params(roth1=1_000_000.0, curve=((0, 1.0), (95, 0.7)))
        flat = simulate(p, 5_000.0, use_curve=False).rows
        curved = simulate(p, 5_000.0, use_curve=True).rows
        self.assertAlmostEqual(flat[-1]["spend_real"], 60_000.0)
        self.assertLess(curved[-1]["spend_real"], flat[-1]["spend_real"])


# ===========================================================================
# Engine basics
# ===========================================================================
class EngineTests(unittest.TestCase):
    def test_zero_everything_gives_simple_division(self):
        # Strategy and order pinned explicitly so a default change cannot silently
        # change what this closed-form test measures.
        p = make_params(nominal_return=0.0, inflation=0.0, tax_enabled=False, trad1=1_200_000.0,
                        strategy="fixed_order", fixed_order=("trad",))
        self.assertAlmostEqual(solve_max_w(p), 1_200_000.0 / (12 * p.n_years), delta=0.05)

    def test_all_roth_matches_annuity_formula(self):
        p = make_params(roth1=1_000_000.0, strategy="fixed_order", fixed_order=("roth",))
        rr = (1 + p.nominal_return) / (1 + p.inflation) - 1
        annual = 1_000_000.0 / sum((1 + rr) ** (-k) for k in range(p.n_years))
        self.assertAlmostEqual(solve_max_w(p), annual / 12, delta=0.05)

    def test_after_tax_spending_equals_need(self):
        p = make_params(trad1=2_000_000.0, ss_fra_monthly1=3_000.0)
        res = simulate(p, 6_000.0)
        self.assertTrue(res.success)
        for r in res.rows:
            net = (r["gross_withdrawal"] + r["ss_gross"] - r["tax_federal"]
                   - r["penalty_early_withdrawal"] - r["surplus_to_cash"])
            self.assertAlmostEqual(net, r["spend_nominal"], delta=ROW_CENT_TOL)

    def test_solved_w_is_the_edge(self):
        p = make_params(trad1=900_000.0, roth1=200_000.0, taxable_value=300_000.0,
                        taxable_basis=200_000.0, cash=50_000.0, ss_fra_monthly1=2_500.0)
        w = solve_max_w(p)
        res = simulate(p, w)
        self.assertTrue(res.success)
        self.assertFalse(simulate(p, w + 5.0).success)
        # "about zero left", scale-invariant; >= 0 (not > 0) because feasibility
        # allows up to SUCCESS_TOLERANCE of rounding at the edge.
        self.assertGreaterEqual(res.final_nominal, 0.0)
        self.assertLess(res.final_nominal, 0.05 * p.total_start_balance)

    def test_taxes_reduce_w(self):
        p_tax = make_params(trad1=1_500_000.0)
        self.assertLess(solve_max_w(p_tax), solve_max_w(replace(p_tax, tax_enabled=False)))

    def test_cost_basis_reduces_tax(self):
        # Large enough that gains exceed the 0% long-term-gains bracket.
        hi = make_params(taxable_value=6_000_000.0, taxable_basis=6_000_000.0)
        lo = make_params(taxable_value=6_000_000.0, taxable_basis=1_000_000.0)
        self.assertGreater(solve_max_w(hi), solve_max_w(lo))
        # Mechanism: same spending, first year -- the low-basis case pays capital-gains tax.
        r_hi, r_lo = simulate(hi, 30_000.0).rows[0], simulate(lo, 30_000.0).rows[0]
        self.assertEqual(r_hi["tax_federal"], 0.0)
        self.assertGreater(r_lo["tax_federal"], 0.0)
        self.assertLess(r_hi["tax_federal"], r_lo["tax_federal"])

    # ---- withdrawal order ----
    def test_fixed_order_draws_in_requested_order(self):
        base = dict(trad1=1_000_000.0, roth1=1_000_000.0)
        roth_first = simulate(make_params(**base, fixed_order=("roth", "trad")), 6_000.0).rows[0]
        trad_first = simulate(make_params(**base, fixed_order=("trad", "roth")), 6_000.0).rows[0]
        self.assertGreater(roth_first["draw_roth"], 0.0)
        self.assertEqual(roth_first["draw_trad"], 0.0)
        self.assertEqual(roth_first["tax_federal"], 0.0)          # Roth money is untaxed
        self.assertGreater(trad_first["draw_trad"], 0.0)
        self.assertEqual(trad_first["draw_roth"], 0.0)

    def test_trad_first_beats_roth_first_when_social_security_follows(self):
        # Mechanism: drawing traditional money BEFORE benefits start keeps it from
        # stacking on top of taxable Social Security later. (Observed gap ~ $96/mo.)
        base = dict(trad1=1_000_000.0, roth1=1_000_000.0, ss_fra_monthly1=4_000.0)
        w_trad_first = solve_max_w(make_params(**base, fixed_order=("trad", "roth")))
        w_roth_first = solve_max_w(make_params(**base, fixed_order=("roth", "trad")))
        self.assertGreater(w_trad_first, w_roth_first + 50.0)

    def test_order_nearly_irrelevant_without_social_security(self):
        # With equal returns on every account and no SS, order should barely matter.
        base = dict(trad1=1_000_000.0, roth1=1_000_000.0)
        a = solve_max_w(make_params(**base, fixed_order=("trad", "roth")))
        b = solve_max_w(make_params(**base, fixed_order=("roth", "trad")))
        self.assertLess(abs(a - b), 10.0)

    # ---- purchases ----
    def test_purchases_reduce_w(self):
        p = make_params(roth1=1_000_000.0, purchases=((2030, 40_000.0, "car"),))
        self.assertLess(solve_max_w(p), solve_max_w(p, include_purchases=False))

    def test_include_purchases_false_with_no_purchases(self):
        p = make_params(roth1=1_000_000.0)
        self.assertAlmostEqual(solve_max_w(p, include_purchases=True),
                               solve_max_w(p, include_purchases=False), places=2)

    def test_purchase_beyond_horizon_is_ignored(self):
        p = make_params(roth1=1_000_000.0, purchases=((2100, 40_000.0, "x"),))
        self.assertAlmostEqual(solve_max_w(p), solve_max_w(p, include_purchases=False), places=2)

    def test_purchase_before_start_is_ignored(self):
        p = make_params(roth1=1_000_000.0, purchases=((2026, 40_000.0, "x"),))
        self.assertAlmostEqual(solve_max_w(p), solve_max_w(p, include_purchases=False), places=2)

    def test_purchase_in_last_year_is_applied(self):
        base = make_params(roth1=1_000_000.0)
        last_year = base.start_year + base.n_years - 1
        with_buy = make_params(roth1=1_000_000.0, purchases=((last_year, 40_000.0, "x"),))
        self.assertLess(solve_max_w(with_buy), solve_max_w(base))

    def test_purchase_in_first_year_is_applied(self):
        with_buy = make_params(roth1=1_000_000.0, purchases=((2027, 40_000.0, "x"),))
        row = simulate(with_buy, 3_000.0).rows[0]
        self.assertAlmostEqual(row["spend_real"], 36_000.0 + 40_000.0, places=2)

    # ---- edge cases ----
    def test_zero_portfolio_returns_zero(self):
        self.assertEqual(solve_max_w(make_params()), 0.0)

    def test_zero_spending_just_compounds(self):
        p = make_params(roth1=1_000_000.0)
        res = simulate(p, 0.0)
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.final_nominal, 1_000_000.0 * 1.05 ** p.n_years, delta=1.0)
        self.assertEqual(res.lifetime_tax, 0.0)

    def test_purchases_exceeding_portfolio_make_w_zero(self):
        # solve_max_w's other early return: even W=0 fails.
        p = make_params(roth1=1_000_000.0, purchases=((2028, 5_000_000.0, "mansion"),))
        self.assertEqual(solve_max_w(p), 0.0)

    # ---- depletion ----
    def test_depletion_year_matches_closed_form(self):
        p = make_params(roth1=1_000_000.0, strategy="fixed_order", fixed_order=("roth",))
        annual_real = 72_000.0                                  # W = 6,000/month
        rr = (1 + p.nominal_return) / (1 + p.inflation) - 1
        b, expected = 1_000_000.0, None
        for k in range(p.n_years):
            if b < annual_real:
                expected = p.start_year + k
                break
            b = (b - annual_real) * (1 + rr)
        self.assertIsNotNone(expected)                          # test setup: it does run out
        res = simulate(p, 6_000.0)
        self.assertFalse(res.success)
        self.assertEqual(res.depletion_year, expected)
        for r in res.rows:
            if r["year"] < res.depletion_year:
                self.assertAlmostEqual(r["shortfall_nominal"], 0.0)
            else:
                self.assertGreater(r["shortfall_nominal"], 0.0)

    def test_no_depletion_year_when_funded(self):
        res = simulate(make_params(roth1=1_000_000.0), 2_000.0)
        self.assertTrue(res.success)
        self.assertIsNone(res.depletion_year)


# ===========================================================================
# Social Security in the engine
# ===========================================================================
class SocialSecurityEngineTests(unittest.TestCase):
    def test_ss_closed_form_with_no_tax_no_growth(self):
        # Born 1965-01-01 -> FRA 67 -> benefits start Jan 2032, run through Dec 2059.
        p = make_params(roth1=1_200_000.0, nominal_return=0.0, inflation=0.0,
                        tax_enabled=False, ss_fra_monthly1=2_000.0,
                        strategy="fixed_order", fixed_order=("roth",))
        by_year = {r["year"]: r for r in simulate(p, 1_000.0).rows}
        # Pin the arithmetic's assumptions instead of trusting them:
        self.assertAlmostEqual(by_year[2031]["ss_gross"], 0.0)
        self.assertAlmostEqual(by_year[2032]["ss_gross"], 24_000.0, places=2)   # 12 months
        self.assertAlmostEqual(by_year[2059]["ss_gross"], 24_000.0, places=2)
        months = (2059 * 12 + 11) - (2032 * 12) + 1                             # 336
        self.assertEqual(months, 336)
        expected = (1_200_000.0 + 2_000.0 * months) / (12 * p.n_years)
        self.assertAlmostEqual(solve_max_w(p), expected, delta=0.05)

    def test_ss_raises_w(self):
        p = make_params(trad1=1_500_000.0)
        self.assertGreater(solve_max_w(replace(p, ss_fra_monthly1=3_000.0)), solve_max_w(p))

    def test_ss_taxed_when_income_is_high(self):
        p = make_params(trad1=2_000_000.0, ss_fra_monthly1=3_000.0)
        late = [r for r in simulate(p, 8_000.0).rows if r["year"] >= 2033]
        self.assertTrue(any(r["ss_taxable"] > 0 for r in late))
        self.assertTrue(all(r["ss_taxable"] <= 0.85 * r["ss_gross"] + 0.01 for r in late))

    def test_ss_benefit_multiplier(self):
        p = make_params(roth1=500_000.0, ss_fra_monthly1=2_000.0)
        full = simulate(p, 1_000.0).rows[10]["ss_gross"]
        cut = simulate(replace(p, ss_multiplier=0.77), 1_000.0).rows[10]["ss_gross"]
        self.assertGreater(full, 0.0)
        self.assertAlmostEqual(cut, 0.77 * full, places=2)

    def test_surplus_ss_goes_to_cash(self):
        p = make_params(roth1=100_000.0, ss_fra_monthly1=5_000.0)
        row = [r for r in simulate(p, 1_000.0).rows if r["year"] == 2032][0]
        self.assertGreater(row["surplus_to_cash"], 0.0)
        self.assertGreater(row["end_cash"], 0.0)


# ===========================================================================
# 59.5 rule and penalties
# ===========================================================================
class EarlyWithdrawalTests(unittest.TestCase):
    # Born 1975-01-01: turns 59.5 in Jul 2034. July is after the June 30 cutoff of
    # the mid-year convention, so access is penalty-free from 2035; the last
    # penalized year is 2034. Age 52 in 2027.
    def young(self, **kw):
        return make_params(person1_birth=date(1975, 1, 1), **kw)

    def test_locked_traditional_and_roth_both_penalized(self):
        # Small Roth runs out first (early_access_order = roth, trad), so over the
        # locked years we see Roth-only, mixed, and trad-only draws.
        p = self.young(trad1=500_000.0, roth1=60_000.0)
        res = simulate(p, 2_000.0)
        by_year = {r["year"]: r for r in res.rows}
        locked_rows = [by_year[y] for y in range(2027, 2035)]
        for r in locked_rows:
            self.assertAlmostEqual(r["penalty_early_withdrawal"],
                                   0.10 * (r["draw_trad"] + r["draw_roth"]), delta=0.02)
            self.assertEqual(r["early_withdrawal_flag"], "PENALTY")
        self.assertTrue(any(r["draw_roth"] > 0 for r in locked_rows))
        self.assertTrue(any(r["draw_trad"] > 0 for r in locked_rows))
        self.assertTrue(any(r["draw_roth"] > 0 and r["draw_trad"] > 0 for r in locked_rows))
        self.assertGreater(res.lifetime_penalty, 0.0)

    def test_lock_boundary_year(self):
        p = self.young(trad1=500_000.0)
        res = simulate(p, 2_000.0)
        by_year = {r["year"]: r for r in res.rows}
        self.assertEqual(by_year[2034]["early_withdrawal_flag"], "PENALTY")   # last locked year
        self.assertEqual(by_year[2035]["early_withdrawal_flag"], "")          # first unlocked year
        self.assertEqual(by_year[2035]["penalty_early_withdrawal"], 0.0)
        self.assertEqual(res.penalty_years, list(range(2027, 2035)))
        self.assertEqual(p.unlock_year(1), 2035)

    def test_no_penalty_when_other_money_exists(self):
        p = self.young(trad1=500_000.0, cash=400_000.0)
        res = simulate(p, 2_000.0)
        first = res.rows[0]
        self.assertEqual(res.penalty_years, [])
        self.assertAlmostEqual(first["end_trad1"], 500_000.0 * 1.05, delta=0.01)  # untouched
        self.assertGreater(first["draw_cash"], 0.0)

    def test_locked_is_per_owner(self):
        # Person 2 is older (unlocked), person 1 is young (locked). Only person 2's
        # account should be drawn in the early years, with no penalty.
        p = self.young(person2_birth=date(1960, 1, 1), trad1=500_000.0, trad2=500_000.0)
        res = simulate(p, 2_000.0)
        self.assertEqual(res.penalty_years, [])
        self.assertAlmostEqual(res.rows[0]["end_trad1"], 500_000.0 * 1.05, delta=0.01)
        self.assertLess(res.rows[0]["end_trad2"], 500_000.0 * 1.05)

    def test_penalty_raises_cost(self):
        locked = self.young(trad1=1_000_000.0)
        # Same account, but the owner is already past 59.5; horizon_age is raised
        # by the 15-year birth-year difference so the number of years is identical.
        unlocked = make_params(person1_birth=date(1960, 1, 1), trad1=1_000_000.0,
                               horizon_age=locked.horizon_age + 15)
        self.assertEqual(locked.n_years, unlocked.n_years)       # setup sanity check
        self.assertLess(solve_max_w(locked), solve_max_w(unlocked))

    def test_tax_disabled_disables_penalty(self):
        p = self.young(trad1=500_000.0, tax_enabled=False)
        self.assertEqual(simulate(p, 2_000.0).lifetime_penalty, 0.0)


# ===========================================================================
# Strategies
# ===========================================================================
class StrategyTests(unittest.TestCase):
    def test_taxable_std_then_cash_draw_amounts(self):
        # basis == value so there are no gains and taxes don't distort the draw split
        p = make_params(cash=1_000_000.0, taxable_value=1_000_000.0,
                        taxable_basis=1_000_000.0, strategy="taxable_std_then_cash")
        row = simulate(p, 8_000.0).rows[0]
        std_nominal = 32_200.0 * 1.025              # 2027 is one year of indexing past 2026
        self.assertAlmostEqual(row["draw_taxable"], std_nominal, places=2)
        self.assertAlmostEqual(row["draw_cash"], 96_000.0 - std_nominal, places=2)

    def test_taxable_std_then_cash_after_cash_runs_out(self):
        p = make_params(cash=10_000.0, taxable_value=2_000_000.0,
                        taxable_basis=2_000_000.0, strategy="taxable_std_then_cash")
        res = simulate(p, 6_000.0)      # sustainable for a $2M portfolio over 33 years
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.rows[3]["draw_cash"], 0.0, places=2)
        self.assertGreater(res.rows[3]["draw_taxable"], 32_200.0 * 1.025 ** 4)

    def test_trad_std_then_cash_fills_deduction_with_traditional(self):
        p = make_params(cash=500_000.0, trad1=1_000_000.0, strategy="trad_std_then_cash")
        row = simulate(p, 6_000.0).rows[0]
        self.assertAlmostEqual(row["draw_trad"], 32_200.0 * 1.025, places=2)
        self.assertEqual(row["tax_federal"], 0.0)   # exactly the standard deduction

    def test_strategies_do_not_touch_locked_money_first(self):
        for name in model.STRATEGIES:
            with self.subTest(strategy=name):
                p = make_params(person1_birth=date(1975, 1, 1), cash=200_000.0,
                                taxable_value=200_000.0, taxable_basis=200_000.0,
                                trad1=800_000.0, strategy=name)
                res = simulate(p, 2_000.0)
                self.assertAlmostEqual(res.rows[0]["end_trad1"], 800_000.0 * 1.05, delta=0.01)


# ===========================================================================
# Cross-cutting invariants (the highest-leverage tests)
# ===========================================================================
def _scenario_configs():
    """Representative households: unlocked, locked (penalty years), and a couple."""
    common = dict(cash=200_000.0, taxable_value=300_000.0, taxable_basis=150_000.0,
                  trad1=400_000.0, roth1=100_000.0, ss_fra_monthly1=2_000.0)
    return {
        "unlocked": dict(common, person1_birth=date(1965, 1, 1)),
        "locked": dict(common, person1_birth=date(1975, 1, 1)),
        "couple": dict(common, person1_birth=date(1975, 1, 1), person2_birth=date(1968, 1, 1),
                       horizon_person="younger", trad2=250_000.0, roth2=50_000.0,
                       ss_fra_monthly2=1_500.0),
    }


class InvariantTests(unittest.TestCase):
    @slow
    def test_all_strategies_satisfy_accounting_identity(self):
        """gross + SS - tax - penalty - surplus + shortfall = spending, every row,
        for every strategy x tax on/off x household. Also checks nothing goes
        negative. (Row-level tolerance is cents: rows are rounded.)"""
        for strategy in model.STRATEGIES:
            for tax_on in (True, False):
                for label, cfg in _scenario_configs().items():
                    p = make_params(strategy=strategy, tax_enabled=tax_on, **cfg)
                    for r in simulate(p, 5_000.0).rows:
                        msg = f"{strategy} tax={tax_on} {label} year={r['year']}"
                        net = (r["gross_withdrawal"] + r["ss_gross"] - r["tax_federal"]
                               - r["penalty_early_withdrawal"] - r["surplus_to_cash"]
                               + r["shortfall_nominal"])
                        self.assertAlmostEqual(net, r["spend_nominal"], delta=ROW_CENT_TOL, msg=msg)
                        for key in ("draw_cash", "draw_taxable", "draw_trad", "draw_roth",
                                    "tax_federal", "penalty_early_withdrawal",
                                    "surplus_to_cash", "shortfall_nominal"):
                            self.assertGreaterEqual(r[key], 0.0, msg=f"{msg} {key}")
                        for key in ("end_cash", "end_taxable", "end_trad1", "end_trad2",
                                    "end_roth1", "end_roth2"):
                            self.assertGreaterEqual(r[key], -0.01, msg=f"{msg} {key}")

    def test_year_level_identity_to_float_precision(self):
        """Same identity at full float precision, by calling the per-year solver
        directly (output rows are rounded, so they cannot show 1e-6)."""
        rng = random.Random(7)
        for trial in range(60):
            p = make_params(
                person1_birth=rng.choice([date(1965, 1, 1), date(1975, 1, 1)]),
                tax_enabled=rng.choice([True, False]),
                strategy=rng.choice(list(model.STRATEGIES)),
                cash=rng.uniform(0, 300_000), taxable_value=rng.uniform(0, 400_000),
                taxable_basis=rng.uniform(0, 400_000), trad1=rng.uniform(0, 600_000),
                roth1=rng.uniform(0, 300_000), ss_fra_monthly1=rng.choice([0.0, 2_500.0]))
            year = rng.choice([2027, 2033, 2040])
            k = year - p.start_year
            price = (1 + p.inflation) ** k
            scale = (1 + p.inflation) ** (year - taxes.BASE_YEAR)
            state = {"cash": p.cash, "taxable": p.taxable_value, "trad1": p.trad1,
                     "trad2": p.trad2, "roth1": p.roth1, "roth2": p.roth2}
            locked = p.locked_buckets(year)
            steps = model._year_steps(p, locked, scale)
            ss = model._ss_income(p, year, price)
            need = rng.uniform(20_000, 150_000) * price
            draws, yt, pen, trad_inc, gains, surplus, short = model._solve_year(
                need, state, p.taxable_basis, ss, scale, p, steps, locked)
            lhs = sum(draws.values()) + ss - yt.income_tax - pen - surplus + short
            self.assertAlmostEqual(lhs, need, delta=1e-6, msg=f"trial {trial}")

    @slow
    def test_shortfall_and_final_balance_are_monotone_in_w(self):
        """The solver's bisection is only valid if more spending never helps.
        Guardrail: any strategy change that breaks this will make solve_max_w lie."""
        rng = random.Random(42)
        for trial in range(40):
            p = make_params(
                person1_birth=rng.choice([date(1965, 1, 1), date(1975, 1, 1)]),
                cash=rng.uniform(0, 300_000), taxable_value=rng.uniform(0, 300_000),
                taxable_basis=rng.uniform(0, 300_000), trad1=rng.uniform(0, 500_000),
                roth1=rng.uniform(0, 500_000), ss_fra_monthly1=rng.choice([0.0, 2_000.0]),
                strategy=rng.choice(list(model.STRATEGIES)))
            prev_short = prev_final = None
            for w in (1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0):
                res = simulate(p, w)
                if prev_short is not None:
                    self.assertGreaterEqual(res.shortfall_total + 1e-6, prev_short,
                                            msg=f"trial {trial}: shortfall fell at w={w}")
                    if res.success and prev_final is not None:
                        self.assertLessEqual(res.final_nominal, prev_final + 1e-6,
                                             msg=f"trial {trial}: final balance rose at w={w}")
                prev_short = res.shortfall_total
                prev_final = res.final_nominal if res.success else None


# ===========================================================================
# Golden regression
# ===========================================================================
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "golden", "representative_scenario.json")


def _golden_run():
    """One scenario touching: couple, SS for both, penalties, three accounts types,
    two purchases, spending curve, and the taxable_std_then_cash strategy."""
    p = make_params(
        person1_birth=date(1972, 6, 15), person2_birth=date(1974, 9, 2),
        horizon_person="younger", cash=40_000.0, taxable_value=250_000.0,
        taxable_basis=150_000.0, trad1=700_000.0, trad2=500_000.0, roth1=100_000.0,
        roth2=60_000.0, ss_fra_monthly1=3_200.0, ss_fra_monthly2=1_800.0,
        strategy="taxable_std_then_cash",
        purchases=((2030, 40_000.0, "car"), (2040, 25_000.0, "trip")),
        curve=((60.0, 1.05), (75.0, 1.0), (85.0, 0.85)))
    res = simulate(p, 6_000.0, use_curve=True)
    return {
        "summary": {"final_nominal": res.final_nominal, "final_real": res.final_real,
                    "shortfall_total": res.shortfall_total, "lifetime_tax": res.lifetime_tax,
                    "lifetime_penalty": res.lifetime_penalty,
                    "penalty_years": res.penalty_years, "depletion_year": res.depletion_year},
        "rows": res.rows,
    }


class GoldenRegressionTests(unittest.TestCase):
    def test_representative_scenario_matches_golden_file(self):
        actual = json.loads(json.dumps(_golden_run()))       # normalize via JSON round-trip
        if os.getenv("UPDATE_GOLDEN"):
            os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
            with open(GOLDEN_PATH, "w") as f:
                json.dump(actual, f, indent=1, sort_keys=True)
            self.skipTest(f"golden file rewritten: {GOLDEN_PATH}")
        self.assertTrue(os.path.exists(GOLDEN_PATH),
                        "golden file missing; run with UPDATE_GOLDEN=1 to create it")
        with open(GOLDEN_PATH) as f:
            golden = json.load(f)
        self.assertEqual(len(actual["rows"]), len(golden["rows"]))
        for key, want in golden["summary"].items():
            got = actual["summary"][key]
            if isinstance(want, (int, float)) and not isinstance(want, bool):
                self.assertAlmostEqual(got, want, delta=0.05, msg=f"summary {key}")
            else:
                self.assertEqual(got, want, msg=f"summary {key}")
        for got_row, want_row in zip(actual["rows"], golden["rows"]):
            self.assertEqual(set(got_row), set(want_row), msg="row columns changed")
            for key, want in want_row.items():
                got = got_row[key]
                if isinstance(want, (int, float)):
                    self.assertAlmostEqual(got, want, delta=0.011,
                                           msg=f"year {want_row['year']} column {key}")
                else:
                    self.assertEqual(got, want, msg=f"year {want_row['year']} column {key}")


if __name__ == "__main__":
    unittest.main()
