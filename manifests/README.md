# File integrity

`SHA256SUMS.txt` records the SHA256 of every released file except itself and ignored runtime outputs/caches. It includes code, configurations, input data, final results and figures.

From the repository root, check the scientific snapshot and all listed files:

```bash
python tools/verify_results.py --hashes
```

The default command without `--hashes` checks scientific counts, sequence coordinates, summary consistency and the presence of final figures. Python 3.11 or newer is required; no third-party packages are needed for this checker.
