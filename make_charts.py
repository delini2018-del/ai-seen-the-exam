"""All charts for the contamination article."""
import glob, gzip, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
B, O, R, G, P = "#2563eb", "#ea580c", "#dc2626", "#6b7280", "#7c3aed"
def foot(fig, s=""): fig.text(0.01, -0.03, "Source: author's experiment on Fashion-MNIST. " + s + "Code: see repo.", fontsize=8.5, color=G)
runs = {(tuple(d["hidden"]), d["leak_fraction"], d["copies"]): d for d in (json.load(open(f)) for f in glob.glob("results/run_*.json"))}
CL = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 1 headline: published vs real
sc = [("No leak", 0.0, 1), ("Leaked\nonce", 1.0, 1), ("Leaked\n5 times", 1.0, 5)]
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, h, name in zip(axs, [(64,), (512, 256)], ["Small model (64 units)", "Larger model (512-256 units)"]):
    x = np.arange(3)
    pub = [100 * runs[(h, f, c)]["benchmark_acc"] for _, f, c in sc]; real = [100 * runs[(h, f, c)]["fresh_acc"] for _, f, c in sc]
    b1 = ax.bar(x - .2, pub, .4, color=O, label="Published score (leaked benchmark)"); b2 = ax.bar(x + .2, real, .4, color=B, label="Real score (never-seen images)")
    for bb in list(b1) + list(b2): ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + .3, f"{bb.get_height():.1f}%", ha="center", fontsize=10)
    ax.set(xticks=x, ylim=(80, 100), title=name); ax.set_xticklabels([s for s, _, _ in sc], fontsize=9.5)
axs[0].set_ylabel("Accuracy (%)"); axs[0].legend(frameon=False, loc="upper left", fontsize=10)
fig.suptitle("The benchmark score rose. The real skill did not.", fontsize=14); foot(fig, "5,000-image benchmark vs 5,000 fresh images. "); fig.savefig("charts/01_published_vs_real.png"); plt.close()

# 2 partial leaks
fig, ax = plt.subplots(figsize=(9, 5))
for h, c, col, lab in [((64,), 1, B, "Small model, leaked once"), ((64,), 5, O, "Small model, leaked 5 times"), ((512, 256), 1, P, "Larger model, leaked once")]:
    pts = sorted((f, d["inflation_pts"]) for (hh, f, cc), d in runs.items() if hh == h and cc == c)
    ax.plot([100 * f for f, _ in pts], [v for _, v in pts], marker="o", lw=2.5, color=col, label=lab)
ax.axhspan(-1.3, 1.3, color=G, alpha=.15); ax.text(2, 1.5, "normal random noise between two test sets (±1.3 pts)", color=G, fontsize=10)
ax.axhline(0, color=G, lw=.8)
ax.set(xlabel="Share of the benchmark leaked into training (%)", ylabel="Inflation: published minus real (pts)", title="Small leaks hide inside the noise")
ax.legend(frameon=False, loc="upper left", fontsize=10); foot(fig); fig.savefig("charts/02_partial_leaks.png"); plt.close()

# 3 per-item loss: leaked vs unleaked
perm = np.random.default_rng(0).permutation(10000); bench = perm[:5000]
L = np.load("results/loss_h64_f0.5_c5.npy"); lk, un = L[bench[:2500]], L[bench[2500:]]
fig, ax = plt.subplots(figsize=(9, 5)); bins = np.logspace(-6, 2, 41)
ax.hist(un, bins, weights=np.ones(len(un)) * 100 / len(un), histtype="step", lw=2.5, color=B, label="Benchmark images never seen")
ax.hist(lk, bins, weights=np.ones(len(lk)) * 100 / len(lk), histtype="step", lw=2.5, color=O, label="Benchmark images leaked 5 times")
ax.set(xscale="log", xlabel="Model's error on the image (log loss, log scale)", ylabel="% of images", title="One by one, leaked images look just like unseen ones")
ax.legend(frameon=False, loc="upper left", fontsize=10); foot(fig, "Small model, half of benchmark leaked 5x. "); fig.savefig("charts/03_loss_overlap.png"); plt.close()
q = [50, 75, 90, 95]; print("loss pct leaked", np.percentile(lk, q).round(4), "unleaked", np.percentile(un, q).round(4))

# 4 audit histogram + 5 near-duplicate gallery
def read(p, o): return np.frombuffer(gzip.open(p).read(), np.uint8, offset=o)
Xtr = read("data/train-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28); Xte = read("data/t10k-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28)
ytr = read("data/train-labels-idx1-ubyte.gz", 8); yte = read("data/t10k-labels-idx1-ubyte.gz", 8)
nn, dist = np.load("results/nn_idx.npy"), np.load("results/nn_dist.npy")
fig, axs = plt.subplots(2, 5, figsize=(11, 5.2))
for k, i in enumerate(np.argsort(dist)[:5]):
    axs[0, k].imshow(Xte[i], cmap="gray_r"); axs[0, k].set_title(f"test #{i}\n{CL[yte[i]]}", fontsize=10)
    axs[1, k].imshow(Xtr[nn[i]], cmap="gray_r"); axs[1, k].set_title(f"train #{nn[i]}\ndistance {dist[i]:.2f}", fontsize=10)
for a in axs.ravel(): a.axis("off")
fig.suptitle("The five closest test-train pairs in Fashion-MNIST (top: test, bottom: nearest training image)", fontsize=12)
foot(fig, "Euclidean distance on pixels scaled 0-1. "); fig.savefig("charts/04_closest_pairs.png"); plt.close()
a = json.load(open("results/audit.json"))
fig, ax = plt.subplots(figsize=(9, 5))
lab = ["min", "1st pct", "5th pct", "25th pct", "median"]; x = np.arange(5)
ax.bar(x - .2, a["pct_test"], .4, color=O, label="Test image → nearest training image"); ax.bar(x + .2, a["pct_train"], .4, color=B, label="Training image → nearest other training image")
ax.set(xticks=x, xticklabels=lab, ylabel="Pixel distance", title="Test images sit no closer to training data than training images do to each other")
ax.legend(frameon=False, fontsize=10); foot(fig, "All 10,000 test images vs 60,000 training images. "); fig.savefig("charts/05_audit.png"); plt.close()
print("charts done")
