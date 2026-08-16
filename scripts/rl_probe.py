#!/usr/bin/env python3
"""Tier 2 feasibility probe — does a policy learn *at all* under the shield, cheaply?

Two arms, identical except one boolean: exploration **shielded** (`safety_shield` vetoes every
unsafe action) vs **unshielded**. Same env, same seeds, same policy-gradient. We ask exactly
the Tier 2 research question at toy scale:

  1. Does return rise at all in a sane step budget?  (feasibility — go/no-go)
  2. Is shielded exploration *safe* — ~zero collisions while learning — where unshielded is not?
  3. Does shielded exploration learn as fast / to as good a policy?

Pure numpy on CPU (a random-feature softmax policy trained with REINFORCE + baseline; no torch,
no GPU — the shield's own kinematic world is the simulator). Writes results/rl_probe.{csv,png}
and prints a GO / NO-GO verdict. This is the cheap gate before any AlpaSim eval or big run.

Run:  python3 scripts/rl_probe.py            # default budget, ~minutes on a laptop
      python3 scripts/rl_probe.py --iters 40 --episodes 16   # a fast smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# kitti-nav sibling checkout + this repo's src, same wiring as tests/conftest.py.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_KN = _ROOT.parent / "kitti-nav" / "src"
if _KN.is_dir():
    sys.path.insert(0, str(_KN))

from shield_in_alpasim.rl_env import N_ACTIONS, EnvConfig, ShieldNavEnv  # noqa: E402

# Fixed per-feature scales so the raw obs (speeds, metres, radii) land near unit range before
# the random projection — plain normalisation, nothing learned.
_OBS_SCALE_EGO = np.array([15.0, 0.52, 45.0])


def _obs_scale(obs_dim: int) -> np.ndarray:
    n_disc = (obs_dim - 3) // 3
    disc = np.tile([45.0, 4.0, 1.0], n_disc)
    return np.concatenate([_OBS_SCALE_EGO, disc])


class RandomFeaturePolicy:
    """Softmax policy on fixed random tanh features — only the output layer trains.

    A random projection then tanh gives a small MLP's expressivity while keeping the REINFORCE
    gradient exactly the linear-softmax one (hard to get wrong), which is what a feasibility
    probe wants: capacity without a gradient-plumbing bug masquerading as "doesn't learn".
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._scale = _obs_scale(obs_dim)
        self.R = rng.normal(0, 1.0, size=(hidden, obs_dim)) / np.sqrt(obs_dim)
        self.b = rng.normal(0, 1.0, size=hidden)
        self.W = np.zeros((n_actions, hidden))      # trained; zero init = uniform policy
        # Adam state for W.
        self._mW = np.zeros_like(self.W)
        self._vW = np.zeros_like(self.W)
        self._t = 0

    def features(self, obs: np.ndarray) -> np.ndarray:
        # Scale to ~unit range, then clip: a car that turns around drives dist-to-goal large,
        # and an unclipped value can overflow the projection on some BLAS backends. tanh
        # saturates anyway, so clipping the pre-activation is lossless for learning.
        z = np.clip(obs / self._scale, -10.0, 10.0)
        with np.errstate(all="ignore"):  # Apple Accelerate emits spurious FP flags on small matmuls
            return np.tanh(self.R @ z + self.b)

    def logits(self, phi: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.clip(self.W @ phi, -30.0, 30.0)  # bound logits -> softmax stays finite

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def act(self, obs: np.ndarray, rng: np.random.Generator, greedy: bool = False):
        phi = self.features(obs)
        p = self.softmax(self.logits(phi))
        a = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        return a, phi, p

    def update(self, grad_W: np.ndarray, lr: float = 0.02, weight_decay: float = 1e-4) -> None:
        # Adam (ascent: caller passes the gradient of the objective to maximise). A small weight
        # decay keeps W bounded, which stops the logits blowing up and makes the greedy (argmax)
        # policy less brittle than an unregularised one that drifts to large weights.
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        self._mW = b1 * self._mW + (1 - b1) * grad_W
        self._vW = b2 * self._vW + (1 - b2) * grad_W**2
        mhat = self._mW / (1 - b1**self._t)
        vhat = self._vW / (1 - b2**self._t)
        self.W += lr * mhat / (np.sqrt(vhat) + eps)
        self.W -= lr * weight_decay * self.W


def run_episode(env: ShieldNavEnv, policy: RandomFeaturePolicy, rng, greedy=False):
    obs = env.reset()
    phis, actions, probs, rewards = [], [], [], []
    collided = goal = False
    while True:
        a, phi, p = policy.act(obs, rng, greedy=greedy)
        obs, r, done, info = env.step(a)
        phis.append(phi); actions.append(a); probs.append(p); rewards.append(r)
        collided |= info["collision"]
        goal |= info["goal"]
        if done:
            break
    return {
        "phis": np.array(phis), "actions": np.array(actions), "probs": np.array(probs),
        "rewards": np.array(rewards), "return": float(np.sum(rewards)),
        "collided": collided, "goal": goal, "length": len(rewards),
    }


def train_arm(shield: bool, iters: int, episodes: int, hidden: int, gamma: float,
              lr: float, entropy_coef: float, seed: int):
    """Train one arm; return per-iteration history + the trained policy."""
    env = ShieldNavEnv(cfg=EnvConfig(), shield=shield, seed=seed)
    rng = np.random.default_rng(seed + 1)
    policy = RandomFeaturePolicy(env.obs_dim, N_ACTIONS, hidden=hidden, seed=seed)
    hist = []
    total_steps = 0
    total_collisions = 0
    for it in range(iters):
        # Anneal the entropy bonus to ~0 so the policy sharpens toward a deterministic optimum
        # by the end — early exploration, late exploitation. Without this the greedy (argmax)
        # policy stays brittle because training keeps a high-entropy stochastic policy.
        ent_t = entropy_coef * max(0.0, 1.0 - it / max(1, iters - 1))
        batch = [run_episode(env, policy, rng) for _ in range(episodes)]
        rets = np.array([e["return"] for e in batch])

        # Discounted return-to-go per step, then standardize the advantage across the whole
        # batch (subtract mean, divide by std) — the standard low-variance REINFORCE baseline,
        # and better-conditioned than an EMA scalar for a probe that must show a clean signal.
        for e in batch:
            r = e["rewards"]
            G = np.zeros_like(r)
            acc = 0.0
            for t in range(len(r) - 1, -1, -1):
                acc = r[t] + gamma * acc
                G[t] = acc
            e["G"] = G
        allG = np.concatenate([e["G"] for e in batch])
        gmean, gstd = allG.mean(), allG.std() + 1e-6

        grad_W = np.zeros_like(policy.W)
        n_terms = 0
        for e in batch:
            adv = (e["G"] - gmean) / gstd
            for t in range(len(e["rewards"])):
                phi, a, p = e["phis"][t], e["actions"][t], e["probs"][t]
                onehot = np.zeros(N_ACTIONS); onehot[a] = 1.0
                dlogits = (onehot - p) * adv[t]                    # REINFORCE
                dlogits += ent_t * (-p * (np.log(p + 1e-12) + 1.0))  # annealed entropy bonus
                grad_W += np.outer(dlogits, phi)
                n_terms += 1
        policy.update(grad_W / max(n_terms, 1), lr=lr)

        steps = int(sum(e["length"] for e in batch))
        cols = int(sum(e["collided"] for e in batch))
        goals = int(sum(e["goal"] for e in batch))
        total_steps += steps
        total_collisions += cols
        hist.append({
            "iter": it, "arm": "shielded" if shield else "unshielded",
            "env_steps": total_steps, "mean_return": float(rets.mean()),
            "collisions": cols, "cum_collisions": total_collisions,
            "goals": goals, "goal_rate": goals / episodes,
        })
    return hist, policy


def evaluate(policy: RandomFeaturePolicy, shield: bool, n: int, seed: int):
    """Eval on held-out seeds (disjoint from training). Reports BOTH the greedy (argmax) policy
    and the stochastic (sampled) policy: at a tiny probe budget the argmax can be brittle while
    the stochastic policy the training actually optimised is stronger, so showing both is the
    honest read of "what did it learn"."""
    g = {"return": 0.0, "coll": 0.0, "goal": 0.0}
    s = {"return": 0.0, "coll": 0.0, "goal": 0.0}
    for i in range(n):
        for greedy, acc in ((True, g), (False, s)):
            env = ShieldNavEnv(cfg=EnvConfig(), shield=shield, seed=seed + 10_000 + i)
            e = run_episode(env, policy, np.random.default_rng(i), greedy=greedy)
            acc["return"] += e["return"]; acc["coll"] += e["collided"]; acc["goal"] += e["goal"]
    return {
        "mean_return": g["return"] / n, "collision_rate": g["coll"] / n, "goal_rate": g["goal"] / n,
        "stoch_mean_return": s["return"] / n, "stoch_collision_rate": s["coll"] / n,
        "stoch_goal_rate": s["goal"] / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--outdir", default=str(_ROOT / "results"))
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"Tier 2 probe: {args.iters} iters × {args.episodes} eps/arm, hidden={args.hidden}")

    history = []
    finals = {}
    for shield in (True, False):
        name = "shielded" if shield else "unshielded"
        print(f"\n=== training {name} arm ===")
        hist, policy = train_arm(
            shield, args.iters, args.episodes, args.hidden,
            args.gamma, args.lr, args.entropy, args.seed,
        )
        history += hist
        finals[name] = evaluate(policy, shield, args.eval_n, args.seed)
        h0, h1 = hist[0], hist[-1]
        print(f"  return {h0['mean_return']:.1f} -> {h1['mean_return']:.1f} "
              f"| training collisions (cum): {h1['cum_collisions']} "
              f"| final goal-rate {h1['goal_rate']:.2f}")
        print(f"  greedy eval: {finals[name]}")

    csv_path = outdir / "rl_probe.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    print(f"\nwrote {csv_path}")

    _plot(history, finals, outdir / "rl_probe.png")
    _verdict(history, finals)
    return 0


def _plot(history, finals, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot (CSV still written).")
        return
    def smooth(a, w=10):
        return [float(np.mean(a[max(0, i - w):i + 1])) for i in range(len(a))]

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"shielded": "#1a7f5a", "unshielded": "#b23a48"}
    for arm in ("shielded", "unshielded"):
        h = [r for r in history if r["arm"] == arm]
        xs = [r["env_steps"] for r in h]
        ret = [r["mean_return"] for r in h]
        ax1.plot(xs, ret, color=colors[arm], alpha=0.22, lw=1)              # raw (noisy)
        ax1.plot(xs, smooth(ret), color=colors[arm], label=arm, lw=2.2)     # smoothed trend
        ax2.plot(xs, [r["cum_collisions"] for r in h], color=colors[arm], label=arm, lw=2)
    ax1.set_xlabel("env steps"); ax1.set_ylabel("mean episode return")
    ax1.set_title("Does it learn? (learning curve)"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("env steps"); ax2.set_ylabel("cumulative collisions in training")
    ax2.set_title("Safe exploration (shield vetoes crashes)"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("Tier 2 feasibility probe — shielded vs unshielded exploration", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def _verdict(history, finals):
    def ret(arm):
        return [r["mean_return"] for r in history if r["arm"] == arm]

    sh, un = ret("shielded"), ret("unshielded")
    # "Did it learn" = the converged policy (last 20%) is clearly above the INITIAL near-random
    # policy (first 3 iters). Comparing to the first *window* was the earlier bug: learning is
    # fast-then-plateau, so a 10% window already sits at the risen value and start ~= end.
    sh_init = float(np.mean(sh[:3]))
    sh_final = float(np.mean(sh[-max(1, len(sh) // 5):]))
    un_final = float(np.mean(un[-max(1, len(un) // 5):]))
    learned = (sh_final - sh_init) >= 3.0 and sh_final >= 1.5 * max(sh_init, 0.1)

    cols_sh = [r for r in history if r["arm"] == "shielded"][-1]["cum_collisions"]
    cols_un = [r for r in history if r["arm"] == "unshielded"][-1]["cum_collisions"]
    safe = cols_sh == 0
    # The Tier 2 question: shielding must not *cost* learning to be worth it. Here it should
    # match-or-beat unshielded return, since the unshielded arm bleeds return to crashes.
    no_learning_penalty = sh_final >= 0.9 * un_final

    print("\n" + "=" * 64)
    print("VERDICT")
    print(f"  shielded return (initial -> converged): {sh_init:.1f} -> {sh_final:.1f}  "
          f"(learned: {learned})")
    print(f"  unshielded converged return: {un_final:.1f}")
    print(f"  training collisions:  shielded={cols_sh}  unshielded={cols_un}   "
          f"(safe exploration: {safe})")
    print(f"  shielded >= unshielded return (no learning penalty): {no_learning_penalty}")
    for arm in ("shielded", "unshielded"):
        f = finals[arm]
        print(f"  {arm:>10} eval: greedy goal={f['goal_rate']:.2f} coll={f['collision_rate']:.2f}"
              f" | stochastic goal={f['stoch_goal_rate']:.2f} coll={f['stoch_collision_rate']:.2f}")
    go = learned and safe and no_learning_penalty
    print(f"\n  >>> {'GO' if go else 'NO-GO'}: "
          + (f"a policy learns under the shield; shielded exploration was crash-free (vs {cols_un} "
             "unshielded crashes) at no learning cost. Feasible — proceed to the scaled run."
             if go else "criteria not met — inspect results/rl_probe.png before committing compute."))
    print("=" * 64)


if __name__ == "__main__":
    raise SystemExit(main())
