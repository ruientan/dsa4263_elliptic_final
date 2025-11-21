# Elliptic Dataset
The Elliptic dataset is a graph-structured cryptocurrency transaction dataset designed to support illicit activity and fraud detection, and blockchain network analysis. It is a time series graph of over 200k Bitcoin transactions (nodes) with a total value of $6 billion, 234k directed payment flow (edges), and 183 node features, and is the largest labelled cryptocurrency transaction dataset publicly available.

The dataset was constructed using blockchain ledger data compiled by Elliptic directly from a Bitcoin blockchain. The dataset represents a sub-graph of the bitcoin blockchain in a Directed Acyclic Graph (DAG), where the in-degree of a node represents the number of inputs of a transaction, while the our-degree represents the number of outputs of a transaction. Alongside the graph metadata, the dataset categorizes the nodes into three classes: **licit**, **illicit**, and **unknown**. A node is considered **illicit** if the transaction has been created by an entity that is considered fraudulent, such as scam networks, malware, terrorist organisations, ransomware, Ponzi scams, among others.

## data folder
if you want to re-train models, download Elliptic csv files from their google drive: https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l

put these in `data/`:
- AddrTx_edgelist.csv
- TxAddr_edgelist.csv
- txs_classes.csv
- txs_edgelist.csv
- txs_features.csv (663mb warning)

trained models already in `models/` so only download if retraining

## dataset structure
                       ┌────────────────────────────────────┐
                       │        Transaction Classes         │
                       │   (txId → {licit/illicit/unknown}) │
                       └───────────────┬────────────────────┘
                                       │ 1-to-1
                                       ▼
                     ┌──────────────────────────────────────┐
                     │        Transaction Features           │
                     │    (183-dimensional node features)    │
                     └───────────────┬──────────────────────┘
                                     │
          ┌──────────────────────────┼─────────────────────────────┐
          ▼                          ▼                             ▼
┌────────────────────┐     ┌────────────────────┐       ┌────────────────────┐
│ Transaction–Tx Edges│     │ Address→Tx Edges   │       │ Tx→Address Edges   │
│   (payment flows)   │     │ (funding inputs)   │       │ (sending outputs)  │
└────────────────────┘     └────────────────────┘       └────────────────────┘



## setup
```
docker-compose build
docker-compose up

#or to run specific models
docker-compose run --rm fraud-detection python3 run_full_experiments.py --models xgboost graphsage gcn --epochs 200

#or specific configurations
docker-compose run --rm fraud-detection python3 run_full_experiments.py --configs engineered_weight
```

## train / val / test split

- **Train:** timesteps 1–34  
- **Validation:** timesteps 35–41  
- **Test:** timesteps 42+  

No future information leaks into training.  
All scalers are fit **only** on training data.


## output
results saved to `results/` - check summary_(timestamp).txt for readable results

models saved to `models/` as {config}_{model}_model.pt or .pkl where config is baseline_noweight, baseline_weight, engineered_noweight, engineered_weight

best model: tuned engineered_weight xgboost (recall=0.4902, auc-pr=0.6529)


# results
## overall summary
| model | weights | f1 | precision | recall | auc-roc | auc-pr | tp | tn | fp | fn |
|-------|---------|-----|-----------|--------|---------|--------|----|----|----|----|
| xgboost (tuned) | yes | 0.6221 | 0.8511 | **0.4902** | 0.9380 | **0.6529** | 200 | 8398 | 35 | 208 |
| xgboost | yes | 0.5879 | 0.7398 | 0.4877 | 0.8544 | 0.5617 | 199 | 8363 | 70 | 209 |
| ensemble | yes | 0.5920 | 0.8525 | 0.4534 | 0.8505 | 0.5495 | 185 | 8401 | 32 | 223 |
| graphsage | yes | 0.5268 | 0.7020 | 0.4216 | 0.8288 | 0.4878 | 172 | 8360 | 73 | 236 |
| gcn (engineered) | yes | 0.3934 | 0.3767 | 0.4118 | 0.8294 | 0.3991 | 168 | 8155 | 278 | 240 |
| gcn (baseline) | yes | 0.4386 | 0.6000 | 0.3456 | 0.8242 | 0.4454 | 141 | 8339 | 94 | 267 |


## focal loss vs cross entropy
### graphsage
| dataset | loss | f1 | precision | recall | auc-roc |
|---------|------|-----|-----------|--------|---------|
| baseline | ce | 0.536 | 0.860 | 0.390 | 0.834 |
| augmented | focal | 0.512 | 0.756 | 0.387 | 0.811 |
| augmented | ce | 0.510 | 0.870 | 0.360 | 0.809 |
| baseline | focal | 0.425 | 0.884 | 0.279 | 0.830 |

### gcn
| dataset | loss | f1 | precision | recall | auc-roc |
|---------|------|-----|-----------|--------|---------|
| augmented | focal | 0.468 | 0.570 | 0.397 | 0.855 |
| augmented | ce | 0.465 | 0.633 | 0.368 | 0.847 |
| baseline | focal | 0.354 | 0.891 | 0.221 | 0.820 |
| baseline | ce | 0.342 | 0.795 | 0.218 | 0.824 |

## key findings
- XGBoost significantly outperforms all GNN models on recall, AUC-PR, and F1.  
- Engineered features contribute more predictive signal than graph message passing.  
- Weighted Cross-Entropy consistently outperforms Focal Loss on this dataset.  
- SHAP reveals that temporal + degree-based engineered features dominate predictive power.  

---

## Project Structure
```
DSA4213-Assignment3/
│
├── data/                            # raw elliptic++ CSV files
├── models/                          # saved PyTorch + XGBoost models
├── results/                         # all experiment outputs, CSVs, SHAP plots
├── tables/                          # summary table
├── .dockerignore
├── .gitignore
├── data_dictionary.txt
│
├── data_loader.py                   # temporal split + PyG graph construction
│
├── docker-compose.yml               # reproducible GPU-enabled environment
├── Dockerfile                       # container build instructions
│
├── ensemble_graphsage_xgboost.py    # GraphSAGE + XGBoost ensemble
├── feature_eng.py                   # engineered feature generation
├── models.py                        # model classes
│
├── README.md       
├── requirements.txt                 # Dependencies
│
├── results_analysis.ipynb           # result analysis notebook         
├── run_focal_experiments.py         # focal loss experiment runer
├── run_full_experiments.py          # main experiment runner
├── shap_tuned_xgb.py                # SHAP feature importance + 95% selection
├── train_tuned_xgb_selected95.py    # final tuned model training
├── train.py                         # GNN training utilities
├── Transactions_EDA_and_Features.ipynb    # EDA + engineered feature exploration  
└── tune_xgboost.py                  # randomised hyperparameter tuning
```

---
## citation
If you use this code, please cite:

Elliptic Dataset:  
M. Weber, C. Domeniconi, J. Chen, D. H. Chau, "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics," KDD 2019.


