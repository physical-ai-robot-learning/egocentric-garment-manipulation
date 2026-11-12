# xvla_data — flattening sets for X-VLA (2026-11-10/11)

- `flat_to_xvla_relcart20.py`: a DiT-layout S03G split → X-VLA RELCART20 / REL32 set (same rows and videos; action (32, 32)
  relative to the current TCP in the X-VLA axes, state = task-anchor poses + jaws 0, per-sample loss masks, train stats).
- `xvla_cumulative.py`: per-day X-VLA roots → one cumulative root with `meta/episode_day.json` (recent-day sampling) and
  the frozen stats of a reference root.

Run with `PYTHONPATH=src` (uses `laundry_deploy.core.xvla_runner` for the axis convention). Training job:
[`../ray_xvla_umiflat/`](../ray_xvla_umiflat/README.md); status: `docs/status_2026-11-12.md` §3.
