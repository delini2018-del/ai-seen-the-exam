"""Part A: are Fashion-MNIST test images already in the training set? Nearest training neighbour
for each of the 10,000 test images (all 60,000 train images, Euclidean distance on raw pixels)."""
import gzip, json, time
import numpy as np
def read(p, o): return np.frombuffer(gzip.open(p).read(), np.uint8, offset=o)
Xtr = read("data/train-images-idx3-ubyte.gz", 16).reshape(-1, 784).astype(np.float32) / 255
ytr = read("data/train-labels-idx1-ubyte.gz", 8)
Xte = read("data/t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784).astype(np.float32) / 255
yte = read("data/t10k-labels-idx1-ubyte.gz", 8)
t = time.time(); trn = (Xtr ** 2).sum(1)
nn, dist = np.zeros(len(Xte), int), np.zeros(len(Xte), np.float32)
for s in range(0, len(Xte), 500):
    q = Xte[s:s+500]; d2 = (q ** 2).sum(1)[:, None] + trn[None] - 2 * q @ Xtr.T
    nn[s:s+500] = d2.argmin(1); dist[s:s+500] = np.sqrt(np.maximum(d2.min(1), 0))
np.save("results/nn_idx.npy", nn); np.save("results/nn_dist.npy", dist)
# baseline: typical nearest-neighbour distance between two *training* images (sample)
idx = np.random.default_rng(0).choice(len(Xtr), 2000, replace=False); q = Xtr[idx]
d2 = (q ** 2).sum(1)[:, None] + trn[None] - 2 * q @ Xtr.T; d2[np.arange(2000), idx] = np.inf
tt = np.sqrt(np.maximum(d2.min(1), 0))
print("secs", round(time.time() - t))
for th in [0.01, 0.5, 1.0, 1.5, 2.0, 2.5]:
    m = dist <= th; print(f"dist<={th}: {m.sum()} test imgs, label agrees {np.mean(ytr[nn[m]]==yte[m]) if m.any() else 0:.3f}")
print("test->train dist pct", np.percentile(dist, [0, 1, 5, 25, 50]).round(2), "train->train", np.percentile(tt, [0, 1, 5, 25, 50]).round(2))
exact = int((dist < 1e-6).sum()); print("exact duplicates", exact)
json.dump(dict(exact=exact, pct_test=np.percentile(dist, [0,1,5,25,50]).tolist(), pct_train=np.percentile(tt,[0,1,5,25,50]).tolist()), open("results/audit.json","w"))
