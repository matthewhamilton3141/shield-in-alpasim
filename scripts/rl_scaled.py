#!/usr/bin/env python3
"""Tier 2 scaled run — safe RL via shielding, the flagship learning-curve figure.

The probe (`rl_probe.py`) proved the mechanism with a numpy REINFORCE policy. This scales it to
the real thing the go/no-go earned: a proper **MLP actor-critic trained with PPO**, **≥5 seeds
per arm** for error bars, on a harder corridor task. Two arms, identical but for the shield flag:
exploration **shielded** (`safety_shield` vetoes every unsafe action) vs **unshielded**. The
question, with statistics this time: does shielded exploration learn as fast / to as safe a policy
as unshielded — and crucially, is it crash-free while doing so?

Still CPU-only (the shield's kitti_nav model is the sim; the MLP is tiny). Obstacle layouts are
procedural here; pass a layout bank sampled from the 10 curated NuRec scenes (via
`ShieldNavEnv(layouts=...)`) to train on the eval distribution once a box is up — the AlpaSim eval
of the trained policy is the box-dependent final step, deferred.

Writes results/rl_scaled.{csv,png} (mean ± std bands across seeds) and prints the verdict.

Run:  python3 scripts/rl_scaled.py                       # 5 seeds/arm, ~300k steps/seed
      python3 scripts/rl_scaled.py --seeds 3 --steps 150000   # a faster pass
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_KN = _ROOT.parent / "kitti-nav" / "src"
if _KN.is_dir():
    sys.path.insert(0, str(_KN))

from shield_in_alpasim.rl_env import N_ACTIONS, EnvConfig, ShieldNavEnv  # noqa: E402

torch.set_num_threads(2)  # small net; more threads just thrash on this workload

# Harder than the probe's task: more obstacles, a tighter corridor, a further goal — enough to
# make the shielded-vs-unshielded difference a real navigation result, not a toy.
SCALED_ENV = EnvConfig(
    corridor_half_width=4.5, goal_x=40.0, n_obstacles=6,
    obstacle_x_range=(10.0, 38.0), horizon=200, n_nearest=6,
)

# Fixed per-feature obs scales (speeds, metres, radii -> ~unit) before the net.
_OBS_SCALE_EGO = np.array([15.0, 0.52, 40.0], np.float32)


def obs_scale(obs_dim: int) -> torch.Tensor:
    n_disc = (obs_dim - 3) // 3
    disc = np.tile([40.0, 4.5, 1.0], n_disc).astype(np.float32)
    return torch.tensor(np.concatenate([_OBS_SCALE_EGO, disc]))


class ActorCritic(nn.Module):
    """Small shared-trunk MLP: policy logits over the 15 discrete actions + a state value."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.register_buffer("scale", obs_scale(obs_dim))
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        # Orthogonal init with a small policy-head gain — standard PPO init, keeps the initial
        # policy near-uniform and the value head well-scaled.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)

    def forward(self, obs):
        h = self.trunk(obs / self.scale)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_np):
        logits, value = self(torch.as_tensor(obs_np, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a), float(dist.log_prob(a)), float(value)


def collect(env, ac, n_steps, obs, ep_state):
    """Roll `n_steps` transitions, auto-resetting on done. Returns a transition buffer and the
    episode stats (returns, collisions, goals) that finished within the window."""
    buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
    ep_returns, ep_cols, ep_goals = [], [], []
    for _ in range(n_steps):
        a, logp, val = ac.act(obs)
        nxt, r, done, info = env.step(a)
        for k, v in (("obs", obs), ("act", a), ("logp", logp), ("val", val),
                     ("rew", r), ("done", done)):
            buf[k].append(v)
        ep_state["ret"] += r
        ep_state["col"] |= info["collision"]
        ep_state["goal"] |= info["goal"]
        obs = nxt
        if done:
            ep_returns.append(ep_state["ret"]); ep_cols.append(int(ep_state["col"]))
            ep_goals.append(int(ep_state["goal"]))
            obs = env.reset(); ep_state.update(ret=0.0, col=False, goal=False)
    # bootstrap value for the last (possibly non-terminal) state
    with torch.no_grad():
        _, last_val = ac(torch.as_tensor(obs, dtype=torch.float32))
    return buf, obs, float(last_val), (ep_returns, ep_cols, ep_goals)


def gae(rew, val, done, last_val, gamma=0.99, lam=0.95):
    T = len(rew)
    adv = np.zeros(T, np.float32)
    gae_t = 0.0
    for t in range(T - 1, -1, -1):
        next_val = last_val if t == T - 1 else val[t + 1]
        nonterminal = 1.0 - float(done[t])
        delta = rew[t] + gamma * next_val * nonterminal - val[t]
        gae_t = delta + gamma * lam * nonterminal * gae_t
        adv[t] = gae_t
    return adv, adv + np.asarray(val, np.float32)


def ppo_update(ac, opt, buf, last_val, ent_coef, clip=0.2, epochs=4, minibatch=256):
    obs = torch.as_tensor(np.array(buf["obs"]), dtype=torch.float32)
    act = torch.as_tensor(np.array(buf["act"]), dtype=torch.long)
    logp_old = torch.as_tensor(np.array(buf["logp"]), dtype=torch.float32)
    adv, ret = gae(buf["rew"], buf["val"], buf["done"], last_val)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv_t = torch.as_tensor(adv, dtype=torch.float32)
    ret_t = torch.as_tensor(ret, dtype=torch.float32)

    n = len(act)
    idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, n, minibatch):
            mb = idx[s:s + minibatch]
            logits, value = ac(obs[mb])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act[mb])
            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv_t[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[mb]
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * (value - ret_t[mb]).pow(2).mean()
            ent = dist.entropy().mean()
            loss = pi_loss + v_loss - ent_coef * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            opt.step()


def train_seed(shield, seed, total_steps, rollout, ent0, lr):
    torch.manual_seed(seed); np.random.seed(seed)
    env = ShieldNavEnv(cfg=SCALED_ENV, shield=shield, seed=seed)
    ac = ActorCritic(env.obs_dim, N_ACTIONS)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    obs = env.reset()
    ep_state = {"ret": 0.0, "col": False, "goal": False}
    hist, steps, cum_col = [], 0, 0
    n_updates = max(1, total_steps // rollout)
    for u in range(n_updates):
        ent_t = ent0 * max(0.0, 1.0 - u / n_updates)   # anneal exploration
        buf, obs, last_val, (rets, cols, goals) = collect(env, ac, rollout, obs, ep_state)
        ppo_update(ac, opt, buf, last_val, ent_t)
        steps += rollout
        cum_col += int(np.sum(cols))
        hist.append({
            "update": u, "env_steps": steps,
            "mean_return": float(np.mean(rets)) if rets else np.nan,
            "goal_rate": float(np.mean(goals)) if goals else np.nan,
            "cum_collisions": cum_col, "n_episodes": len(rets),
        })
    return hist, ac


def evaluate(ac, shield, n, seed):
    env_seed0 = seed + 100_000
    ret = coll = goal = 0.0
    for i in range(n):
        env = ShieldNavEnv(cfg=SCALED_ENV, shield=shield, seed=env_seed0 + i)
        obs = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                logits, _ = ac(torch.as_tensor(obs, dtype=torch.float32))
            obs, r, done, info = env.step(int(torch.argmax(logits)))
            ret += r; coll += info["collision"]; goal += info["goal"]
    return {"mean_return": ret / n, "collision_rate": coll / n, "goal_rate": goal / n}


def _aggregate(per_seed_hist):
    """Stack per-seed histories on a common update axis -> (steps, mean, std) per metric."""
    m = min(len(h) for h in per_seed_hist)
    steps = np.array([per_seed_hist[0][u]["env_steps"] for u in range(m)])
    out = {"env_steps": steps}
    for key in ("mean_return", "goal_rate", "cum_collisions"):
        arr = np.array([[np.nan_to_num(h[u][key], nan=0.0) for u in range(m)]
                        for h in per_seed_hist], float)
        out[key + "_mean"] = arr.mean(0)
        out[key + "_std"] = arr.std(0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--rollout", type=int, default=4096)
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--outdir", default=str(_ROOT / "results"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"Tier 2 scaled run (PPO): {args.seeds} seeds × {args.steps} steps/arm, "
          f"rollout {args.rollout}")

    agg, finals, rows = {}, {}, []
    for shield in (True, False):
        arm = "shielded" if shield else "unshielded"
        per_seed, evals = [], []
        for sd in range(args.seeds):
            hist, ac = train_seed(shield, sd, args.steps, args.rollout, args.entropy, args.lr)
            per_seed.append(hist)
            evals.append(evaluate(ac, shield, args.eval_n, sd))
            print(f"  [{arm}] seed {sd}: return "
                  f"{hist[0]['mean_return']:.1f} -> {hist[-1]['mean_return']:.1f} | "
                  f"cum_collisions {hist[-1]['cum_collisions']} | eval {evals[-1]}")
            for h in hist:
                rows.append({"arm": arm, "seed": sd, **h})
        agg[arm] = _aggregate(per_seed)
        finals[arm] = {
            k: (float(np.mean([e[k] for e in evals])), float(np.std([e[k] for e in evals])))
            for k in ("mean_return", "collision_rate", "goal_rate")
        }

    with open(outdir / "rl_scaled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {outdir / 'rl_scaled.csv'}")
    _plot(agg, outdir / "rl_scaled.png")
    _verdict(agg, finals)
    return 0


def _plot(agg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"shielded": "#1a7f5a", "unshielded": "#b23a48"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, a in agg.items():
        x = a["env_steps"]
        for ax, key in ((ax1, "mean_return"), (ax2, "cum_collisions")):
            mean, std = a[key + "_mean"], a[key + "_std"]
            ax.plot(x, mean, color=colors[arm], lw=2.2, label=arm)
            ax.fill_between(x, mean - std, mean + std, color=colors[arm], alpha=0.18)
    ax1.set_xlabel("env steps"); ax1.set_ylabel("mean episode return")
    ax1.set_title("Learning curve (PPO, mean ± std over seeds)"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("env steps"); ax2.set_ylabel("cumulative collisions in training")
    ax2.set_title("Safe exploration"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("Tier 2 — shielded vs unshielded exploration (PPO)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def _verdict(agg, finals):
    sh, un = agg["shielded"], agg["unshielded"]
    sh_init = float(sh["mean_return_mean"][:2].mean())
    sh_fin = float(sh["mean_return_mean"][-max(1, len(sh["mean_return_mean"]) // 5):].mean())
    un_fin = float(un["mean_return_mean"][-max(1, len(un["mean_return_mean"]) // 5):].mean())
    cols_sh = float(sh["cum_collisions_mean"][-1])
    cols_un = float(un["cum_collisions_mean"][-1])
    learned = (sh_fin - sh_init) >= 3.0
    safe = cols_sh == 0
    print("\n" + "=" * 64)
    print("VERDICT (PPO, seed-averaged)")
    print(f"  shielded return: {sh_init:.1f} -> {sh_fin:.1f}   unshielded -> {un_fin:.1f}")
    print(f"  training collisions (mean/seed): shielded={cols_sh:.0f}  unshielded={cols_un:.0f}")
    for arm, f in finals.items():
        print(f"  {arm:>10} eval: return {f['mean_return'][0]:.1f}±{f['mean_return'][1]:.1f} "
              f"goal {f['goal_rate'][0]:.2f} coll {f['collision_rate'][0]:.2f}")
    print(f"\n  learned={learned}  safe_exploration={safe}")
    print("=" * 64)


if __name__ == "__main__":
    raise SystemExit(main())
