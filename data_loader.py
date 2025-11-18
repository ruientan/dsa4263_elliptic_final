"""
Data loader for Elliptic Bitcoin transaction dataset
"""
import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class EllipticDataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.scaler = StandardScaler()

    def load_raw_data(self):
        """Load raw CSV files from ellipticpp directory"""
        print("Loading raw data...")

        # Transaction features and labels
        tx_features = pd.read_csv(os.path.join(self.data_dir, 'txs_features.csv'))
        tx_classes = pd.read_csv(os.path.join(self.data_dir, 'txs_classes.csv'))
        tx_edges = pd.read_csv(os.path.join(self.data_dir, 'txs_edgelist.csv'))

        # Address-transaction mappings for feature engineering
        addr_tx_in = pd.read_csv(os.path.join(self.data_dir, 'AddrTx_edgelist.csv'))
        tx_addr_out = pd.read_csv(os.path.join(self.data_dir, 'TxAddr_edgelist.csv'))

        return tx_features, tx_classes, tx_edges, addr_tx_in, tx_addr_out

    def prepare_features(self, tx_features, tx_classes):
        """Merge features with classes and filter unknowns"""
        # Merge features and classes
        df = tx_features.merge(tx_classes, on='txId', how='left')

        # Map classes: 1=illicit, 2=licit, 3=unknown
        df['label'] = df['class'].map({1: 1, 2: 0, 3: -1})  # illicit=1, licit=0, unknown=-1

        # Store original columns for later
        self.time_steps = df['Time step'].values
        self.tx_ids = df['txId'].values
        self.labels_all = df['label'].values

        # Get feature columns (exclude metadata)
        feature_cols = [col for col in df.columns
                       if col not in ['txId', 'Time step', 'class', 'label']]

        # Extract features
        features = df[feature_cols].values.astype(np.float32)

        # Replace any NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features, df

    def build_graph(self, tx_edges, tx_ids):
        """Build graph structure from edges"""
        print("Building graph structure...")

        # Create node ID mapping
        unique_ids = list(tx_ids)
        id_to_idx = {tx_id: idx for idx, tx_id in enumerate(unique_ids)}

        # Filter edges to only include nodes we have
        valid_edges = tx_edges[
            (tx_edges['txId1'].isin(unique_ids)) &
            (tx_edges['txId2'].isin(unique_ids))
        ]

        # Convert to node indices
        edge_index = []
        for _, row in valid_edges.iterrows():
            src = id_to_idx[row['txId1']]
            dst = id_to_idx[row['txId2']]
            edge_index.append([src, dst])
            edge_index.append([dst, src])  # Make undirected

        if len(edge_index) > 0:
            edge_index = np.array(edge_index).T
        else:
            edge_index = np.array([[], []])

        return edge_index, id_to_idx

    def create_masks(self, labels, time_steps):
        """Create train/val/test masks based on time steps"""
        print("Creating temporal train/val/test splits...")

        # Temporal split based on timesteps
        train_mask = (time_steps <= 34) & (labels != -1)  # Exclude unknowns
        val_mask = (time_steps >= 35) & (time_steps <= 41) & (labels != -1)
        test_mask = (time_steps >= 42) & (labels != -1)

        # Print split statistics
        print(f"Train: {train_mask.sum()} samples (timesteps 1-34)")
        print(f"Val: {val_mask.sum()} samples (timesteps 35-41)")
        print(f"Test: {test_mask.sum()} samples (timesteps 42+)")

        # Class distribution
        for name, mask in [('Train', train_mask), ('Val', val_mask), ('Test', test_mask)]:
            if mask.sum() > 0:
                illicit_pct = (labels[mask] == 1).mean() * 100
                print(f"{name} - Illicit: {illicit_pct:.1f}%, Licit: {100-illicit_pct:.1f}%")

        return train_mask, val_mask, test_mask

    def get_pytorch_geometric_data(self, use_engineered=False):
        """Create PyTorch Geometric Data object"""
        # Load data
        tx_features, tx_classes, tx_edges, addr_tx_in, tx_addr_out = self.load_raw_data()

        # Add engineered features if requested
        if use_engineered:
            from feature_eng import FeatureEngineer
            engineer = FeatureEngineer(self.data_dir)
            df, eng_features = engineer.engineer_features(
                tx_features, tx_classes, addr_tx_in, tx_addr_out
            )
            # Use engineered dataframe
            self.time_steps = df['time_step'].values if 'time_step' in df else df['Time step'].values
            self.tx_ids = df['txId'].values
            self.labels_all = df['class'].map({1: 1, 2: 0, 3: -1}).values

            # Get all numeric features except metadata
            exclude_cols = ['txId', 'Time step', 'time_step', 'class', 'label', 'out_range_ratio']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            features = df[feature_cols].select_dtypes(include=[np.number]).values.astype(np.float32)
            print(f"Using {features.shape[1]} features (including engineered)")
        else:
            # Original base features only
            features, df = self.prepare_features(tx_features, tx_classes)

        # Normalize features using training data only
        train_mask_temp = (self.time_steps <= 34) & (self.labels_all != -1)
        self.scaler.fit(features[train_mask_temp])
        features_normalized = self.scaler.transform(features)

        # Build graph
        edge_index, id_to_idx = self.build_graph(tx_edges, self.tx_ids)

        # Create masks
        train_mask, val_mask, test_mask = self.create_masks(self.labels_all, self.time_steps)

        # Convert to PyTorch tensors
        x = torch.FloatTensor(features_normalized)
        y = torch.LongTensor(self.labels_all)
        edge_index = torch.LongTensor(edge_index)

        # Create Data object
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=torch.BoolTensor(train_mask),
            val_mask=torch.BoolTensor(val_mask),
            test_mask=torch.BoolTensor(test_mask)
        )

        # Store additional info
        data.num_features = features.shape[1]
        data.num_classes = 2  # Binary classification
        data.time_steps = torch.LongTensor(self.time_steps)
        data.tx_ids = self.tx_ids

        # Class weights for imbalanced data
        train_labels = y[train_mask]
        class_counts = torch.bincount(train_labels[train_labels >= 0])
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        data.class_weights = class_weights

        print(f"\nGraph statistics:")
        print(f"Nodes: {x.shape[0]}")
        print(f"Edges: {edge_index.shape[1]}")
        print(f"Features: {x.shape[1]}")
        print(f"Class weights: {class_weights.numpy()}")

        return data, df

    def get_feature_matrix_for_xgboost(self, use_engineered=False):
        """Get feature matrix and labels for XGBoost training"""
        # Load data
        tx_features, tx_classes, _, addr_tx_in, tx_addr_out = self.load_raw_data()

        if use_engineered:
            from feature_eng import FeatureEngineer
            engineer = FeatureEngineer(self.data_dir)
            df, eng_features = engineer.engineer_features(
                tx_features, tx_classes, addr_tx_in, tx_addr_out
            )
            # Setup for engineered features
            self.time_steps = df['time_step'].values if 'time_step' in df else df['Time step'].values
            self.tx_ids = df['txId'].values
            self.labels_all = df['class'].map({1: 1, 2: 0, 3: -1}).values

            # Get all numeric features except metadata
            exclude_cols = ['txId', 'Time step', 'time_step', 'class', 'label', 'out_range_ratio']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            features = df[feature_cols].select_dtypes(include=[np.number]).values.astype(np.float32)
            print(f"XGBoost using {features.shape[1]} features (including engineered)")
        else:
            # Prepare features
            features, df = self.prepare_features(tx_features, tx_classes)

        # Filter out unknowns
        known_mask = self.labels_all != -1
        features_known = features[known_mask]
        labels_known = self.labels_all[known_mask]
        time_steps_known = self.time_steps[known_mask]

        # Normalize
        train_mask = time_steps_known <= 34
        self.scaler.fit(features_known[train_mask])
        features_normalized = self.scaler.transform(features_known)

        # Create train/val/test splits
        train_idx = time_steps_known <= 34
        val_idx = (time_steps_known >= 35) & (time_steps_known <= 41)
        test_idx = time_steps_known >= 42

        return {
            'X_train': features_normalized[train_idx],
            'y_train': labels_known[train_idx],
            'X_val': features_normalized[val_idx],
            'y_val': labels_known[val_idx],
            'X_test': features_normalized[test_idx],
            'y_test': labels_known[test_idx]
        }