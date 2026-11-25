"""训练 ckpt → 部署 ckpt 导出 CLI。

用法::

    python -m kick_sim2real.export \
        --src logs/rsl_rl/upper_ukick_cmd/<run>/model_1999.pt \
        --dst ~/kick_policy.pt
"""

from __future__ import annotations

import argparse

from .policy import export_ckpt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="训练侧 rsl_rl ckpt")
    p.add_argument("--dst", required=True, help="输出部署 ckpt（.pt）")
    a = p.parse_args()
    meta = export_ckpt(a.src, a.dst)
    print(f"[kick_sim2real] exported -> {a.dst}")
    print(f"  md5={meta['md5']}  obs_dim={meta['obs_dim']}  "
          f"hidden={meta['hidden_dims']}  iter={meta.get('iter')}")


if __name__ == "__main__":
    main()
