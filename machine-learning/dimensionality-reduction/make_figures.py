"""Generate all reference figures for the dimensionality-reduction & clustering tutorial.

Run from this folder with any python that has numpy + matplotlib + scikit-learn:

    python make_figures.py

Everything is synthetic and seeded - output is deterministic and reproducible.
Images are written to ./images/ and referenced from the markdown files.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs

HERE = Path(__file__).resolve().parent
IMG = HERE / "images"
IMG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.titleweight": "bold"})
rng = np.random.default_rng(42)

# ---------------------------------------------------------------- 01 PCA geometry
X = np.concatenate([rng.normal(0, 1, (200, 1)), rng.normal(0, 0.25, (200, 1))], axis=1)
X = X @ rng.normal(size=(2, 2))                      # rotate arbitrarily
pc = PCA(2).fit(X)
fig, ax = plt.subplots(figsize=(6.2, 5))
ax.scatter(X[:, 0], X[:, 1], s=14, c="C0", alpha=0.6)
mu = pc.mean_
for (v, e, col, name), ls in zip(zip(pc.components_, pc.explained_variance_ratio_,
                                     ["C3", "C1"], ["PC1", "PC2"]), ["-", "--"]):
    ax.quiver(*mu, *(2.6 * np.sqrt(e) * v), angles="xy", scale_units="xy", scale=1,
              color=col, ls=ls, width=0.006, label=f"{name} ({e:.0%} variance)")
ax.scatter(*mu, marker="X", c="k", s=70, zorder=5, label="mean (data is centred first)")
ax.legend(fontsize=8, loc="upper left")
ax.set(title="PCA finds orthogonal axes ordered by variance",
       xlabel="feature 1", ylabel="feature 2")
fig.savefig(IMG / "01_pca_geometry.png"); plt.close(fig)

# ---------------------------------------------------------------- 02 scree / elbow
ev = np.array([0.42, 0.23, 0.12, 0.06, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01])
ev = ev / ev.sum()
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.bar(range(1, 11), ev, color="C0", alpha=0.85, label="per-component ratio")
ax.plot(range(1, 11), np.cumsum(ev), "o-", color="C3", label="cumulative")
ax.axhline(0.9, ls="--", color="0.5")
ax.annotate("90% of variance reached at k=6 -> keep 6 components",
            (6, 0.9), xytext=(6.1, 0.72), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="0.3"))
ax.set(xlabel="component index", ylabel="explained variance ratio",
       title="Scree plot: choose k where the curve flattens (the elbow)")
ax.legend(fontsize=8)
fig.savefig(IMG / "02_pca_scree.png"); plt.close(fig)

# ---------------------------------------------------------------- 03 projecting the mean is exact
d = 40
pts = np.vstack([rng.normal(0, 1, (25, d)) + 2, rng.normal(0, 1, (25, d)) - 2])
pc = PCA(2).fit(pts)
proj = pc.transform(pts)                             # f(x) = W^T (x - mu)
means = proj[[*range(25)], :].mean(0), proj[[*range(25, 50)], :].mean(0)
means_of_pts = np.array([pts[:25].mean(0), pts[25:].mean(0)])
proj_of_means = pc.transform(means_of_pts)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
for g, col, name in [(slice(0, 25), "C0", "group A"), (slice(25, 50), "C3", "group B")]:
    ax.scatter(proj[g, 0], proj[g, 1], s=22, c=col, alpha=0.6, label=name)
for (mx, my), (px, py), col in zip(means, proj_of_means, ["C0", "C3"]):
    ax.scatter(mx, my, marker="*", s=420, c="none", edgecolor="k", lw=1.6, zorder=5)
    ax.scatter(px, py, marker="o", s=130, c=col, edgecolor="k", zorder=6)
ax.legend(fontsize=8)
ax.set(title="Projection of a mean (circle) == mean of projections (star):\n"
             "they coincide because PCA is a linear/affine map",
       xlabel="PC1", ylabel="PC2")
fig.savefig(IMG / "03_pca_mean_exact.png"); plt.close(fig)
# ---------------------------------------------------------------- 04 t-SNE intuition
Xb, yb = make_blobs(n_samples=360, centers=4, cluster_std=[0.8, 1.6, 0.5, 2.2],
                    center_box=(-12, 12), random_state=1)
Pb = PCA(2).fit_transform(Xb)
Tb = TSNE(2, perplexity=30, init="pca", random_state=0,
          learning_rate="auto").fit_transform(Xb)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].scatter(Pb[:, 0], Pb[:, 1], c=yb, cmap="tab10", s=16)
ax[0].set_title("PCA: global distances readable, axes meaningful,\n"
                "but dense clusters shrink / big blobs dominate")
ax[1].scatter(Tb[:, 0], Tb[:, 1], c=yb, cmap="tab10", s=16)
ax[1].set_title("t-SNE: neighbourhoods pulled apart;\n"
                "cluster SIZE and inter-cluster DISTANCE mean almost nothing")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.savefig(IMG / "04_tsne_intuition.png"); plt.close(fig)

# ---------------------------------------------------------------- 05 perplexity
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, perp in zip(ax, [2, 15, 50]):
    T = TSNE(2, perplexity=perp, init="pca", random_state=0,
             learning_rate="auto").fit_transform(Xb)
    a.scatter(T[:, 0], T[:, 1], c=yb, cmap="tab10", s=14)
    a.set_title(f"perplexity = {perp}\n(~effective #neighbours)")
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Same data, different perplexity - t-SNE is a prior, not a measurement", y=1.02)
fig.savefig(IMG / "05_tsne_perplexity.png"); plt.close(fig)

# ---------------------------------------------------------------- 06 no out-of-sample map
Xq, yq = make_blobs(n_samples=240, centers=3, cluster_std=1.1, random_state=3)
query = np.array([[0.2, 6.5]])                       # a point in the gap
fit1 = TSNE(2, perplexity=20, init="pca", random_state=0,
            learning_rate="auto").fit_transform(Xq)
fit2 = TSNE(2, perplexity=20, init="pca", random_state=0,
            learning_rate="auto").fit_transform(np.vstack([Xq, query]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap="tab10", s=16)
ax[0].scatter(*query[0], marker="*", s=420, c="none", edgecolor="k", lw=1.8)
ax[0].set_title("input space: 3 clusters + one new query point (star)")
ax[1].scatter(fit1[:, 0], fit1[:, 1], c=yq, cmap="tab10", s=16, label="t-SNE fit WITHOUT query")
ax[1].scatter(fit2[:len(Xq), 0], fit2[:len(Xq), 1], c=yq, cmap="tab10", s=16,
              marker="x", label="same points, fit WITH query")
ax[1].scatter(*fit2[len(Xq)], marker="*", s=420, c="none", edgecolor="k", lw=1.8)
ax[1].legend(fontsize=8)
ax[1].set_title("output: adding 1 point moved ALL points ->\n"
                "there is no fixed map f(x); every fit is a new coordinate system")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.savefig(IMG / "06_tsne_no_out_of_sample.png"); plt.close(fig)

# ---------------------------------------------------------------- 07 cosine vs L2 geometry
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
a = plt.Circle((0, 0), 1, fill=False, color="0.6", lw=0.8)
ax[0].add_artist(a)
for v, col, name in [((0.9, 0.4), "C0", "a"), ((1.8, 0.8), "C1", "b = 2a"),
                     ((0.4, 0.9), "C3", "c")]:
    ax[0].annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
    ax[0].text(v[0] + 0.05, v[1] + 0.05, name, color=col, fontsize=12, weight="bold")
ax[0].set_xlim(-0.3, 2.2); ax[0].set_ylim(-0.3, 1.4); ax[0].set_aspect("equal")
ax[0].set(title="cos(a,b)=1  (same direction, ||b||=2||a||)\ncos(a,c)=0.73 (same norm, different angle)",
          xlabel="cosine ignores length, sees only the angle")
ax[0].grid(True, alpha=0.3)
th = np.linspace(0, np.pi, 200)
ax[1].plot(th, np.sqrt(2 - 2 * np.cos(th)), "C2", label="L2 on the unit sphere: sqrt(2-2cos t)")
ax[1].plot(th, np.cos(th), "C0", label="cosine similarity: cos t")
ax[1].plot(th, 1 - np.cos(th), "C3--", label="cosine distance: 1 - cos t")
ax[1].set(xlabel="angle between vectors (rad)", ylabel="value",
          title="For L2-normalised vectors L2 and cosine are\nmonotone transforms of each other: L2^2 = 2(1-cos t)")
ax[1].legend(fontsize=8)
fig.savefig(IMG / "07_cosine_vs_l2.png"); plt.close(fig)

# ---------------------------------------------------------------- 08 similarity heatmap
centers = rng.normal(0, 3, (3, 30))
Xh = np.vstack([rng.normal(0, 0.5, (12, 30)) + c for c in centers])
Xn = Xh / np.linalg.norm(Xh, axis=1, keepdims=True)
M = Xn @ Xn.T
order = np.arange(36)
shuf = rng.permutation(36)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.8))
for a, idx, t in [(ax[0], order, "sorted by group: BLOCK structure = clusters"),
                  (ax[1], shuf, "same matrix shuffled: structure is invisible")]:
    im = a.imshow(M[np.ix_(idx, idx)], cmap="magma")
    a.set_title(t, fontsize=9)
    a.set_xticks([]); a.set_yticks([])
    if a is ax[0]:
        for b in (12, 24):
            a.axhline(b - 0.5, color="w", lw=1.2); a.axvline(b - 0.5, color="w", lw=1.2)
fig.colorbar(im, ax=ax, fraction=0.03)
fig.savefig(IMG / "08_similarity_heatmap.png"); plt.close(fig)

# ---------------------------------------------------------------- 09 silhouette anatomy
Xs2, ys2 = make_blobs(n_samples=120, centers=2, cluster_std=1.2,
                     center_box=(-9, 9), random_state=7)
ss = silhouette_samples(Xs2, ys2)
i = np.flatnonzero(ys2 == ys2[0])[10]
same = Xs2[ys2 == ys2[i]]; other = Xs2[ys2 != ys2[i]]
a_i = np.linalg.norm(same - Xs2[i], axis=1).mean()
b_i = np.linalg.norm(other - Xs2[i], axis=1).mean()
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax[0].scatter(Xs2[:, 0], Xs2[:, 1], c=ys2, cmap="coolwarm", s=18, alpha=0.7)
ax[0].scatter(*Xs2[i], c="k", s=60, zorder=5)
for p in same[1:6]:
    ax[0].annotate("", xy=p, xytext=Xs2[i], arrowprops=dict(arrowstyle="-", color="C0", alpha=0.6))
for p in other[:3]:
    ax[0].annotate("", xy=p, xytext=Xs2[i], arrowprops=dict(arrowstyle="-", color="C3", alpha=0.6))
ax[0].set_title(f"for point i (black):\na(i) = mean dist to own cluster (blue) = {a_i:.2f}\n"
                f"b(i) = mean dist to other cluster (red) = {b_i:.2f}\n"
                f"s(i) = (b-a)/max(a,b) = {(b_i - a_i) / max(a_i, b_i):.2f}", fontsize=9)
ax[1].barh(np.arange(len(ss)), np.sort(ss), color="C0")
ax[1].axvline(ss.mean(), color="C3", lw=2, label=f"mean silhouette = {ss.mean():.2f}")
ax[1].set(ylabel="samples (sorted)", xlabel="silhouette s(i)",
          title="silhouette plot: one bar per sample\nbars below 0 = point closer to another cluster")
ax[1].legend(fontsize=8)
fig.savefig(IMG / "09_silhouette_anatomy.png"); plt.close(fig)

# ---------------------------------------------------------------- 10 separation regimes
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
regimes = [
    ("clean clusters\nbig centroid gap, small spread", 2.2, (-4, 4), 0.8),
    ("overlapping clouds\ncentroids apart but spread >= gap", 2.2, (-1.2, 1.2), 2.0),
    ("one blob, arbitrary split\ncentroids close, spread big", 2.2, (0.0, 0.1), 2.5),
]
for a, (t, _, cen, sd) in zip(ax, regimes):
    pts, ys = [], []
    for k, cx in enumerate(cen):                    # two isotropic Gaussians
        pts.append(rng.normal(0, sd, (50, 2)) + np.array([cx, 0]))
        ys += [k] * 50
    X3, y3 = np.vstack(pts), np.array(ys)
    a.scatter(X3[:, 0], X3[:, 1], c=y3, cmap="coolwarm", s=16, alpha=0.75)
    for k in (0, 1):
        m = X3[y3 == k].mean(axis=0)
        a.scatter(*m, marker="X", c="k", s=90, zorder=5)
    c0, c1 = (X3[y3 == k].mean(axis=0) for k in (0, 1))
    gap = np.linalg.norm(c1 - c0)
    spread = 0.5 * (np.linalg.norm(X3[y3 == 0] - c0, axis=1).mean()
                    + np.linalg.norm(X3[y3 == 1] - c1, axis=1).mean())
    a.set_title(f"{t}\ncentroid gap={gap:.1f}  mean radius={spread:.1f}\n"
                f"silhouette={silhouette_score(X3, y3):.2f}", fontsize=9)
    a.set_xticks([]); a.set_yticks([])
fig.savefig(IMG / "10_separation_regimes.png"); plt.close(fig)

print("all figures written to", IMG)
for p in sorted(IMG.glob("*.png")):
    print("  ", p.name)