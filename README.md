# The AI Had Already Seen the Exam

Code for the Medium article of the same name: a controlled test-set contamination experiment on Fashion-MNIST (CPU only, seeded).

The 10,000 test images are split once into a 5,000-image **benchmark** (the score you publish) and a 5,000-image **fresh** set that never leaks. A share of the benchmark is copied into training data (1× or 5×), and published vs real accuracy are compared.

| Result | Number |
|---|---|
| Larger model (512-256), benchmark leaked 5×: published vs real | 97.1% vs 88.2% |
| Same model, no leak: benchmark vs fresh | 87.6% vs 88.2% |
| Small model, 25% of benchmark leaked once: inflation | +0.5 pts (normal noise ≈ ±1.3 pts) |
| Per-image detection of leaked items (AUROC) | 0.47–0.52 |
| Real audit: test→train near-duplicates vs within-train | 5/16/82 vs 7/16/82 at three thresholds |

## Run it
```bash
pip install -r requirements.txt
bash run_all.sh
```

## Files
- `audit.py` – nearest training neighbour for every test image + within-train baseline
- `leak_run.py HIDDEN LEAK_FRACTION COPIES EPOCHS` – one contamination run, writes `results/run_*.json`
- `make_charts.py` – every chart in the article
- `confidence_lab.py` – data loading helpers

## Limits
Small image classifiers with deliberate leaks, not LLMs. Results illustrate how contamination behaves; they do not measure any specific model's contamination.

## Credits
Fashion-MNIST © Zalando SE, MIT License. Code: MIT License.
