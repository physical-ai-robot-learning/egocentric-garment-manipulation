"""Figure: human vs robotized vs executed TCP (top view) and robotization displacement, demo 2 of take 20261021_171057, run 200335.
Reads only; writes PDF to the report figures dir. Also prints the numbers quoted in the text."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/jeonghwanlee/ego_collector/datasets/"
E = D + "tag_rig/20261021_171057/"
RUN = D + "replay/bi_20261021_200335/"
OUT = sys.argv[1]
segs = json.load(open(E + "segments.json")); g = [x for x in segs if x["rep"] == 2][0]
t0, t1 = g["hold_pre"][0], g["hold_post"][1]
BLUE, ORANGE, GRAY, TXT = "#2a78d6", "#eb6834", "#8a8984", "#52514e"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": TXT, "xtick.color": TXT, "ytick.color": TXT,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.0, 1.35]))
for s, ls in (("right", "-"), ("left", "-")):
    z = np.load(E + f"umi_traj_{s}_fused_robot.npz", allow_pickle=True); t = z["t_ns"].astype(np.int64); m = (t >= t0) & (t <= t1)
    H, Rb, d, q = z["p_B_human"][m], z["p_B"][m], z["robotize_disp_m"][m], z["robotize_quality"][m]
    r = np.load(RUN + f"{s}/seg_00.npz"); mt = json.load(open(RUN + f"{s}/meta.json")); Tr = np.array(mt["T_ready"])
    C = r["p_obs"] + Tr[:3, 3] + r["start_offset"]; B = r["p_cmd"] + Tr[:3, 3] + r["start_offset"]
    k = slice(len(r["p_cmd"]))  # executed during the stream only (C has a 1 s tail)
    lab = s == "right"
    ax[0].plot(H[:, 1] * 1e3, H[:, 0] * 1e3, color="#c9c8c2", lw=2.6, label="human A" if lab else None)
    ax[0].plot(Rb[:, 1] * 1e3, Rb[:, 0] * 1e3, color=BLUE, lw=1.0, label="robotized A$'$" if lab else None)
    ax[0].plot(C[k][:, 1] * 1e3, C[k][:, 0] * 1e3, color=ORANGE, lw=1.0, ls=(0, (2, 1.2)), label="robot C" if lab else None)
    tt = (t[m] - t[m][0]) / 1e9
    ax[1].plot(tt, d * 1e3, color=BLUE if s == "right" else ORANGE, lw=1.0, label=f"{s} hand")
    print(s, f"frames {m.sum()}, disp median {np.median(d)*1e3:.1f} p95 {np.percentile(d,95)*1e3:.1f} max {d.max()*1e3:.1f} mm, "
          f"quality 0/1/2 = {np.mean(q==0)*100:.0f}/{np.mean(q==1)*100:.0f}/{np.mean(q==2)*100:.0f} %, "
          f"y human p1/p99 {np.percentile(H[:,1],1)*1e3:.0f}/{np.percentile(H[:,1],99)*1e3:.0f} -> {np.percentile(Rb[:,1],1)*1e3:.0f}/{np.percentile(Rb[:,1],99)*1e3:.0f} mm, "
          f"zmin human {H[:,2].min()*1e3:.1f} -> {Rb[:,2].min()*1e3:.1f} mm")
ax[0].invert_xaxis(); ax[0].set_xlabel("$y_B$ (mm, +left)"); ax[0].set_ylabel("$x_B$ (mm, forward)"); ax[0].set_aspect("equal", adjustable="datalim")
ax[0].axvline(0, color="#d6d5cf", lw=0.6, zorder=0); ax[0].legend(frameon=False, loc="lower center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, -0.42))
ax[0].set_title("(a) top view, robot base frame", fontsize=7, color=TXT)
for y_, l_ in ((20, "20 mm"), (50, "50 mm")):
    ax[1].axhline(y_, color="#b5b4ae", lw=0.6, ls="--", zorder=0); ax[1].text(0.3, y_ + 1.5, l_, fontsize=6, color=TXT)
ax[1].set_xlabel("demo time (s)"); ax[1].set_ylabel("$\\|p_{A'}-p_A\\|$ (mm)"); ax[1].legend(frameon=False, fontsize=6, loc="upper right")
ax[1].set_title("(b) robotization displacement per frame", fontsize=7, color=TXT)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
