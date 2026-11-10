r"""Hybrid quantum policy: a variational quantum circuit chooses the MILP's daily risk knobs on Chronos-2 forecasts.

The classical baseline is Chronos-2 + the MILP with the knobs Optuna picked on Q3 (fixed every day). Here a policy
reads the day's 16 gate-A numbers (qest.gym_env.day_features) and sets kappa_up, kappa_dn, tol per day. The policy
is a data re-uploading circuit (Perez-Salinas et al., Quantum 4, 226, 2020; for RL Jerbi et al., NeurIPS 2021 and
Skolik et al., Quantum 6, 720, 2022; for trading Gurgul, Chen & Lessmann, arXiv:2601.18811, 2026): 8 qubits, 3 layers
of [RY(w_in x) encoding with trainable input weights, trainable RZ RY, CZ ring], read out as <Z_i>, then linear actor
and critic heads (hybrid). Trained by PPO (Stable-Baselines3) on Q3 days, the actor's bias set to the Optuna knobs so
it starts at the baseline and learns when to deviate; Q4 scored once with the deterministic policy.
usage: .venv/bin/python -m qest.qrl [--steps 2048] [--envs 8]   -> results/rl_qpolicy.csv
"""
import argparse
import json
import math

import numpy as np
import torch
from torch import nn

NQ, LAYERS = 8, 3


class VQC(nn.Module):
    """obs (B, 16) -> <Z_i> (B, 8) of the re-uploading circuit, simulated exactly (2^8 amplitudes)."""

    def __init__(self, n_in=16):
        super().__init__(); g = torch.Generator().manual_seed(0); self.n_in = n_in
        self.w = nn.Parameter(.1 * torch.randn(LAYERS, NQ, 2, generator=g)); self.win = nn.Parameter(torch.ones(LAYERS, NQ))
        N = 1 << NQ; bits = (torch.arange(N)[:, None] >> torch.arange(NQ)) & 1
        cz = torch.ones(N)
        for q in range(NQ):
            cz = cz * torch.where((bits[:, q] & bits[:, (q + 1) % NQ]).bool(), -1.0, 1.0)
        self.register_buffer('cz', cz.to(torch.complex64)); self.register_buffer('z', (1 - 2 * bits).float())

    @staticmethod
    def _gate(psi, q, U):
        B = psi.shape[0]; v = psi.reshape(B, 1 << (NQ - q - 1), 2, 1 << q)
        return torch.einsum('bij,bajc->baic', U, v).reshape(B, -1)

    @staticmethod
    def _ry(t):
        c, s = torch.cos(t / 2), torch.sin(t / 2); return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)

    @staticmethod
    def _rz(t):
        e = torch.exp(-.5j * t.to(torch.complex64)); z = torch.zeros_like(e); return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)

    def forward(self, obs):
        x = math.pi * torch.tanh(obs.float()); B = len(x)
        psi = torch.zeros(B, 1 << NQ, dtype=torch.complex64, device=x.device); psi[:, 0] = 1
        for l in range(LAYERS):
            for q in range(NQ):
                feat = x[:, (l * NQ + q) % self.n_in]  # layer l re-uploads the next 8 of the 16 inputs, cycling
                psi = self._gate(psi, q, self._ry(self.win[l, q] * feat))
                U = (self._ry(self.w[l, q, 1]) @ self._rz(self.w[l, q, 0])).expand(B, 2, 2)
                psi = self._gate(psi, q, U)
            psi = psi * self.cz
        return (psi.abs() ** 2) @ self.z


def extractor():
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class QFeatures(BaseFeaturesExtractor):
        def __init__(self, space):
            super().__init__(space, NQ); self.vqc = VQC(space.shape[0])

        def forward(self, obs):
            return self.vqc(obs)
    return QFeatures


def knobs():
    from qest.paths import RESULTS
    B = json.loads((RESULTS / 'tune_best_milp_chronos.json').read_text()); P = B['params']
    a0 = np.array([P['kappa_up'], P['kappa_dn'], math.log(P['tol'] / .01) / math.log(30)]); return B['margins'], np.clip(a0, 0, 1)


def make(split, sequential=False):
    from qest.gym_env import KratTradeEnv
    M, _ = knobs(); return KratTradeEnv(split, sequential=sequential, kind='chronos', margins=M)


def main():
    import pandas as pd
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from qest.paths import RESULTS
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=2048); ap.add_argument('--envs', type=int, default=8)
    a = ap.parse_args(); torch.manual_seed(0); M, a0 = knobs()
    env = make_vec_env(lambda: make('val'), n_envs=a.envs, seed=0, vec_env_cls=SubprocVecEnv)
    model = PPO('MlpPolicy', env, n_steps=16, batch_size=64, n_epochs=10, learning_rate=3e-3, gamma=0.0, seed=0, verbose=0,
                policy_kwargs=dict(features_extractor_class=extractor(), net_arch=[], log_std_init=-1.5))
    with torch.no_grad():  # start at the baseline: the actor's mean = the Optuna knobs, weights small
        model.policy.action_net.weight.mul_(.01); model.policy.action_net.bias.copy_(torch.tensor(a0, dtype=torch.float32))
    model.learn(total_timesteps=a.steps); env.close()
    rows = []
    for split in ('val', 'test'):
        e = make(split, sequential=True)
        for _ in e.days:
            obs, info = e.reset(); act, _ = model.predict(obs, deterministic=True); _, r, _, _, inf = e.step(act)
            act = np.clip(act, 0, 1); rows.append(dict(split=split, day=inf['day'], eur=inf['eur'], kappa_up=act[0], kappa_dn=act[1], tol=.01 * 30 ** act[2]))
    R = pd.DataFrame(rows); R.to_csv(RESULTS / 'rl_qpolicy.csv', index=False); model.save(RESULTS / 'rl_qpolicy_model')
    print(R.groupby('split')[['eur']].sum().round(0).to_string()); print(R.groupby('split')[['kappa_up', 'kappa_dn', 'tol']].agg(['mean', 'std']).round(3).to_string())


def _check():
    q = VQC(); o = q(torch.randn(5, 16)); assert o.shape == (5, NQ) and o.abs().max() <= 1 + 1e-5
    o.sum().backward(); assert q.w.grad.abs().sum() > 0; print('qrl check ok')


if __name__ == '__main__':
    import sys
    _check() if '--check' in sys.argv else main()
