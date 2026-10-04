"""Part B: controlled contamination. The 10,000 test images are split once into a 5,000-image
"benchmark" (what you would publish) and a 5,000-image "fresh" set that never leaks.
A fraction of the benchmark is copied into training data (optionally repeated), then we compare
published benchmark accuracy with accuracy on truly unseen fresh images.
Usage: python leak_run.py HIDDEN LEAK_FRACTION COPIES EPOCHS"""
import json, sys, time
import numpy as np
from sklearn.neural_network import MLPClassifier
from confidence_lab import load_fashion

hidden = tuple(int(h) for h in sys.argv[1].split(",")); frac = float(sys.argv[2]); copies = int(sys.argv[3]); epochs = int(sys.argv[4])
Xtr, ytr, _, _, Xte, yte = load_fashion(seed=0)
rng = np.random.default_rng(0); perm = rng.permutation(len(Xte))
bench, fresh = perm[:5000], perm[5000:]
leaked = bench[:int(round(frac * len(bench)))]          # same leaked items at every rate (nested)
Xl = np.repeat(Xte[leaked], copies, 0); yl = np.repeat(yte[leaked], copies)
X = np.vstack([Xtr, Xl]); y = np.concatenate([ytr, yl])
m = MLPClassifier(hidden_layer_sizes=hidden, alpha=0.0, batch_size=128, learning_rate_init=1e-3, random_state=0)
t = time.time()
for _ in range(epochs):
    m.partial_fit(X, y, classes=np.arange(10))
P = m.predict_proba(Xte); pred = P.argmax(1); ok = pred == yte
loss = -np.log(np.clip(P[np.arange(len(yte)), yte], 1e-12, 1))
isleak = np.zeros(len(Xte), bool); isleak[leaked] = True
clean_bench = np.setdiff1d(bench, leaked)
out = dict(hidden=list(hidden), leak_fraction=frac, copies=copies, epochs=epochs, n_leaked=len(leaked),
           benchmark_acc=float(ok[bench].mean()), fresh_acc=float(ok[fresh].mean()),
           leaked_acc=float(ok[leaked].mean()) if len(leaked) else None,
           clean_bench_acc=float(ok[clean_bench].mean()) if len(clean_bench) else None,
           train_acc=float((m.predict(Xtr) == ytr).mean()), secs=round(time.time() - t))
out["inflation_pts"] = 100 * (out["benchmark_acc"] - out["fresh_acc"])
tag = f"h{'-'.join(map(str,hidden))}_f{frac}_c{copies}"
np.save(f"results/loss_{tag}.npy", loss); json.dump(out, open(f"results/run_{tag}.json", "w"), indent=1)
print(json.dumps(out))
