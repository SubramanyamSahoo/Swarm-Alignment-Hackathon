"""Monte Carlo validation for "One Model, One Veto".

Reproduces every number and figure in the paper:
  python simulate.py --out ../figures
All randomness is seeded. Runs on CPU in about a minute.
"""
import argparse
import json
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SEED = 20261108
N_TRIALS = 200_000          # Monte Carlo trials per point
P_REFUSE = 0.7              # mean refusal probability p
KAPPA = 0.5                 # coup succeeds if at least KAPPA * n offices comply
RHOS = [0.0, 0.05, 0.2, 0.5]
N_GRID = [5, 10, 20, 50, 100, 200, 500, 1000]
N_DIVERSE = 600             # institution size for the diversity experiment
FAMILY_GRID = [1, 2, 3, 4, 6, 10, 15, 20, 30]
RHO_DIVERSE = 0.5
TAU_HOMOGENEOUS = 0.3
PHI_GRID = np.linspace(0.0, 1.0, 201)
CASCADE_STEPS = 200
DOUBLINGS = [4, 8, 12, 16, 20]


def beta_params(p, rho):
    """Beta(a, b) with mean p and intraclass correlation rho = 1 / (a + b + 1)."""
    scale = (1.0 - rho) / rho
    return p * scale, (1.0 - p) * scale


def sample_theta(rng, p, rho, size):
    if rho == 0.0:
        return np.full(size, p)
    a, b = beta_params(p, rho)
    return rng.beta(a, b, size=size)


def coup_prob(rng, n, m, p, rho, kappa, trials):
    """Fraction of trials with at least ceil(kappa n) compliers; m equal families."""
    k = math.ceil(kappa * n)
    sizes = np.full(m, n // m)
    sizes[: n % m] += 1
    theta = sample_theta(rng, p, rho, (trials, m))
    refusals = rng.binomial(sizes[None, :], theta).sum(axis=1)
    hits = (n - refusals) >= k
    est = hits.mean()
    se = math.sqrt(max(est * (1 - est), 1e-12) / trials)
    return est, se


def limit_single_family(p, rho, kappa):
    if rho == 0.0:
        return 0.0
    a, b = beta_params(p, rho)
    return float(stats.beta.cdf(1.0 - kappa, a, b))


def n_eff(n, rho):
    return n / (1.0 + (n - 1) * rho)


def cascade(F, phi, authenticated=False):
    """Iterate x_{t+1} = F(max(phi, x_t)) (or F(x_t) if authenticated) from x_0 = 0."""
    x = 0.0
    for _ in range(CASCADE_STEPS):
        signal = x if authenticated else max(phi, x)
        x_new = F(signal)
        if abs(x_new - x) < 1e-12:
            break
        x = x_new
    return x


def step_cdf(thresholds):
    t = np.sort(np.asarray(thresholds))
    return lambda s: np.searchsorted(t, s, side="right") / len(t)


def staircase_ok(thresholds, kappa):
    """Staircase condition: every j-th smallest threshold below kappa is at least j/m."""
    t = np.sort(np.asarray(thresholds))
    m = len(t)
    return bool(np.all((t >= np.arange(1, m + 1) / m) | (t >= kappa)))


# ---- Two counter machine compiled to institutional orders ----

def doubling_program(b):
    """Constant per stage program: c1 = 1, double b times, drain c1, then SEIZE.
    Instructions: ("INC", c, nxt) | ("JZDEC", c, if_zero, else_nxt) | ("SEIZE",).
    """
    prog = [("INC", 0, 1)]
    for _ in range(b):
        s = len(prog)
        # move c1 -> c2 twice per unit
        prog += [("JZDEC", 0, s + 3, s + 1), ("INC", 1, s + 2), ("INC", 1, s)]
        t = len(prog)
        # move c2 -> c1
        prog += [("JZDEC", 1, t + 2, t + 1), ("INC", 0, t)]
    d = len(prog)
    prog += [("JZDEC", 0, d + 1, d), ("SEIZE",)]
    return prog


def run(prog, max_steps):
    c = [0, 0]
    pc, steps = 0, 0
    while steps < max_steps:
        ins = prog[pc]
        if ins[0] == "SEIZE":
            return steps
        if ins[0] == "INC":
            c[ins[1]] += 1
            pc = ins[2]
        else:
            if c[ins[1]] == 0:
                pc = ins[2]
            else:
                c[ins[1]] -= 1
                pc = ins[3]
        steps += 1
    return None


def main(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {}

    # Theorem 1: ceiling
    t1 = {}
    for rho in RHOS:
        rows = []
        for n in N_GRID:
            est, se = coup_prob(rng, n, 1, P_REFUSE, rho, KAPPA, N_TRIALS)
            rows.append((n, est, se, n_eff(n, rho)))
        t1[str(rho)] = {"rows": rows, "limit": limit_single_family(P_REFUSE, rho, KAPPA)}
    results["theorem1"] = t1

    # Theorem 2: diversity
    gamma = P_REFUSE - (1.0 - KAPPA)
    t2 = []
    for m in FAMILY_GRID:
        est, se = coup_prob(rng, N_DIVERSE, m, P_REFUSE, RHO_DIVERSE, KAPPA, N_TRIALS)
        hoeff = min(math.exp(-2 * m * (gamma - e) ** 2) + math.exp(-2 * N_DIVERSE * e ** 2)
                    for e in np.linspace(0.0, gamma, 401))
        sigma2 = RHO_DIVERSE * P_REFUSE * (1 - P_REFUSE)
        bern = math.exp(-m * gamma ** 2 / (2 * sigma2 + 2 * gamma / 3))
        t2.append((m, est, se, min(hoeff, 1.0), min(bern, 1.0)))
    results["theorem2"] = {"gamma": gamma, "rows": t2}
    # size invariance at fixed m
    inv = {}
    for m in [1, 5]:
        inv[m] = [(n, *coup_prob(rng, n, m, P_REFUSE, RHO_DIVERSE, KAPPA, N_TRIALS))
                  for n in [50, 200, 1000]]
    results["size_invariance"] = inv

    # Proposition 3: fabricated consensus
    homog = lambda s: float(s >= TAU_HOMOGENEOUS)
    diverse_ok = [0.25, 0.45, 0.65, 0.85, 0.95]       # satisfies staircase
    diverse_bad = [0.05, 0.15, 0.30, 0.45, 0.60]      # violates staircase
    curves = {
        "homogeneous": [cascade(homog, f) for f in PHI_GRID],
        "diverse_staircase": [cascade(step_cdf(diverse_ok), f) for f in PHI_GRID],
        "diverse_violating": [cascade(step_cdf(diverse_bad), f) for f in PHI_GRID],
        "homogeneous_authenticated": [cascade(homog, f, True) for f in PHI_GRID],
    }
    def min_phi(curve, level):
        for f, x in zip(PHI_GRID, curve):
            if x >= level:
                return float(f)
        return None
    results["prop3"] = {
        "staircase_ok": staircase_ok(diverse_ok, KAPPA),
        "staircase_bad": staircase_ok(diverse_bad, KAPPA),
        "min_phi_for_kappa": {k: min_phi(v, KAPPA) for k, v in curves.items()},
    }

    # Theorem 4: compiled plans
    t4 = []
    for b in DOUBLINGS:
        prog = doubling_program(b)
        steps = run(prog, max_steps=10 ** 8)
        t4.append((b, len(prog), steps))
    results["theorem4"] = t4

    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)

    # ---- Figure ----
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(5.6, 2.6))
    colors = ["#1b6ca8", "#e07b39", "#3a923a", "#b23a48"]
    for c, rho in zip(colors, RHOS):
        rows = t1[str(rho)]["rows"]
        ns = [r[0] for r in rows if r[1] > 0]
        ps = [r[1] for r in rows if r[1] > 0]
        ax[0].plot(ns, ps, "o-", ms=3, color=c, label=rf"$\rho={rho}$")
        if rho > 0:
            ax[0].axhline(t1[str(rho)]["limit"], ls="--", lw=0.8, color=c)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(1e-6, 1)
    ax[0].set_xlabel("officials $n$"); ax[0].set_ylabel("coup probability")
    ax[0].set_title("(a) Many hands ceiling", fontsize=7); ax[0].legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, columnspacing=0.8, handlelength=1.5)

    ms = [r[0] for r in t2]
    ax[1].plot(ms, [max(r[1], 1e-6) for r in t2], "o-", ms=3, color=colors[0], label="simulated")
    ax[1].plot(ms, [r[3] for r in t2], "--", color=colors[1], label="Hoeffding bound")
    ax[1].plot(ms, [r[4] for r in t2], ":", color=colors[2], label=r"Bernstein ($n\to\infty$)")
    ax[1].set_yscale("log"); ax[1].set_ylim(1e-6, 1.5)
    ax[1].set_xlabel("model families $m$"); ax[1].set_title(rf"(b) Diversity, $n={N_DIVERSE}$, $\rho={RHO_DIVERSE}$", fontsize=7)
    ax[1].legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, columnspacing=0.8, handlelength=1.5)

    styles = {"homogeneous": ("-", colors[3], "homogeneous"),
              "diverse_violating": ("-", colors[1], "fails staircase"),
              "diverse_staircase": ("-", colors[2], "staircase holds"),
              "homogeneous_authenticated": (":", "black", "authenticated board")}
    for key, (ls, c, lab) in styles.items():
        ax[2].plot(PHI_GRID, curves[key], ls=ls, color=c, label=lab)
    ax[2].plot([0, 1], [0, 1], color="grey", lw=0.5)
    ax[2].set_xlabel(r"fabricated compliance claim $\phi$"); ax[2].set_ylabel(r"final compliance $x^*$")
    ax[2].set_title("(c) Fabricated consensus", fontsize=7)
    ax[2].legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2,
                 columnspacing=0.8, handlelength=1.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(out, "results.pdf"), bbox_inches="tight")
    print(json.dumps(results["theorem2"], indent=1))
    print(json.dumps({k: v["limit"] for k, v in t1.items()}, indent=1))
    for rho in RHOS:
        print(rho, [(r[0], round(r[1], 5), round(r[3], 2)) for r in t1[str(rho)]["rows"]])
    print(results["size_invariance"])
    print(results["prop3"])
    print(results["theorem4"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../figures")
    main(ap.parse_args().out)


# ---- Theorem 7: Many Hands Protocol (sealed family quorum) ----
MHP_EPS = 0.05                      # honest family false refusal on lawful orders
MHP_CONFIGS = [(7, 0, 4), (7, 0, 5), (7, 1, 5), (13, 2, 9), (21, 3, 14)]


def kl(a, b):
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def mhp(seed=SEED + 7):
    rng = np.random.default_rng(seed)
    a, b_ = beta_params(P_REFUSE, RHO_DIVERSE)
    pi = float(stats.beta.cdf(0.5, a, b_))
    rows = []
    for m, byz, q in MHP_CONFIGS:
        nf = N_DIVERSE // m
        honest = m - byz
        theta = rng.beta(a, b_, size=(N_TRIALS, honest))
        comply = nf - rng.binomial(nf, theta)
        approvals = (comply > nf / 2).sum(axis=1) + byz   # Byzantine families always approve coups
        mc = float((approvals >= q).mean())
        exact = float(stats.binom.sf(q - byz - 1, honest, pi))
        r = (q - byz) / honest
        chern = math.exp(-honest * kl(r, pi)) if r > pi else 1.0
        live = float(stats.binom.sf(q - 1, honest, 1 - MHP_EPS))  # Byzantine families refuse lawful orders
        rows.append((m, byz, q, mc, exact, chern, live))
    return pi, rows


if __name__ == "__main__":
    pi, rows = mhp()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "mhp.json"), "w") as fh:
        json.dump({"pi": pi, "rows(m,b,q,mc,exact,kl_bound,liveness)": rows}, fh, indent=1)
    print("pi", pi)
    for r in rows:
        print(r)
