## data folder
if you want to re-train models, download elliptic++ csv files from their google drive: https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l

put these in `data/`:
- AddrTx_edgelist.csv
- TxAddr_edgelist.csv
- txs_classes.csv
- txs_edgelist.csv
- txs_features.csv (663mb warning)

trained models already in `models/` so only download if retraining

## setup
```
docker-compose build
docker-compose up

#or to run specific models
docker-compose run --rm fraud-detection python3 run_full_experiments.py --models xgboost graphsage --epochs 200

#or specific configurations
docker-compose run --rm fraud-detection python3 run_full_experiments.py --configs engineered_weight
```

## output
results saved to `results/` - check summary_(timestamp).txt for readable results

models saved to `models/` as {config}_{model}_model.pt or .pkl where config is baseline_noweight, baseline_weight, engineered_noweight, engineered_weight

best model: baseline_noweight xgboost (f1=0.612)


# results

## baseline features (elliptic++ original)
| model | weights | f1 | precision | recall | auc-roc | auc-pr | time (s) | tp | tn | fp | fn |
|-------|---------|-----|-----------|--------|---------|--------|----------|----|----|----|----|
| xgboost | no | **0.612** | 0.866 | 0.473 | 0.860 | 0.559 | 1.7 | 193 | 8403 | 30 | 215 |
| xgboost | yes | 0.598 | 0.780 | 0.485 | 0.853 | 0.562 | 2.5 | 198 | 8377 | 56 | 210 |
| graphsage | no | 0.528 | 0.797 | 0.395 | 0.836 | 0.500 | 6.9 | 161 | 8392 | 41 | 247 |
| graphsage | yes | 0.525 | 0.642 | 0.444 | 0.855 | 0.488 | 5.2 | 181 | 8332 | 101 | 227 |
| gcn | yes | 0.438 | 0.455 | 0.422 | 0.826 | 0.431 | 3.4 | 172 | 8227 | 206 | 236 |
| gat | yes | 0.355 | 0.293 | 0.449 | 0.856 | 0.365 | 11.8 | 183 | 7992 | 441 | 225 |
| gat | no | 0.351 | 0.393 | 0.316 | 0.803 | 0.267 | 8.7 | 129 | 8234 | 199 | 279 |
| gcn | no | 0.342 | 0.795 | 0.218 | 0.824 | 0.408 | 4.2 | 89 | 8410 | 23 | 319 |

## engineered features (baseline + eda)
| model | weights | f1 | precision | recall | auc-roc | auc-pr | time (s) | tp | tn | fp | fn |
|-------|---------|-----|-----------|--------|---------|--------|----------|----|----|----|----|
| xgboost | no | 0.605 | 0.839 | 0.473 | 0.863 | 0.559 | 1.8 | 193 | 8396 | 37 | 215 |
| xgboost | yes | 0.563 | 0.681 | 0.480 | 0.850 | 0.556 | 2.6 | 196 | 8341 | 92 | 212 |
| graphsage | no | 0.535 | 0.704 | 0.431 | 0.826 | 0.452 | 7.9 | 176 | 8359 | 74 | 232 |
| graphsage | yes | 0.502 | 0.544 | 0.466 | 0.814 | 0.445 | 4.9 | 190 | 8274 | 159 | 218 |
| gcn | no | 0.471 | 0.601 | 0.387 | 0.840 | 0.469 | 5.1 | 158 | 8328 | 105 | 250 |
| gcn | yes | 0.410 | 0.367 | 0.466 | 0.852 | 0.450 | 3.3 | 190 | 8105 | 328 | 218 |
| gat | no | 0.337 | 0.328 | 0.346 | 0.801 | 0.261 | 5.7 | 141 | 8144 | 289 | 267 |
| gat | yes | 0.312 | 0.227 | 0.500 | 0.814 | 0.448 | 11.9 | 204 | 7739 | 694 | 204 |

## focal loss vs cross entropy
### graphsage
| dataset | loss | f1 | precision | recall | auc-roc |
|---------|------|-----|-----------|--------|---------|
| baseline | ce | 0.536 | 0.860 | 0.390 | 0.834 |
| augmented | focal | **0.512** | 0.756 | 0.387 | 0.811 |
| augmented | ce | 0.510 | 0.870 | 0.360 | 0.809 |
| baseline | focal | 0.425 | 0.884 | 0.279 | 0.830 |

### gcn
| dataset | loss | f1 | precision | recall | auc-roc |
|---------|------|-----|-----------|--------|---------|
| augmented | focal | 0.468 | 0.570 | 0.397 | 0.855 |
| augmented | ce | 0.465 | 0.633 | 0.368 | 0.847 |
| baseline | focal | 0.354 | 0.891 | 0.221 | 0.820 |
| baseline | ce | 0.342 | 0.795 | 0.218 | 0.824 |

### gat
| dataset | loss | f1 | precision | recall | auc-roc |
|---------|------|-----|-----------|--------|---------|
| augmented | ce | 0.378 | 0.407 | 0.353 | 0.798 |
| baseline | ce | 0.372 | 0.443 | 0.321 | 0.810 |
| augmented | focal | 0.335 | 0.313 | 0.360 | 0.811 |
| baseline | focal | 0.314 | 0.380 | 0.267 | 0.795 |