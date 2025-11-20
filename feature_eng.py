import pandas as pd
import numpy as np
import os

class FeatureEngineer:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir

    def engineer_features(self, tx_features, tx_classes, addr_tx_in, tx_addr_out):
        df = tx_features.merge(tx_classes, on='txId', how='left')

        rename_dict = {
            'Time step': 'time_step',
            'in_txs_degree': 'in_degree',
            'out_txs_degree': 'out_degree'
        }

        for old_col, new_col in rename_dict.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        print("Adding engineered features...")

        # Address patterns
        fan_out = tx_addr_out.groupby('txId')['output_address'].nunique().reset_index()
        fan_out.columns = ['txId', 'fan_out']

        fan_in = addr_tx_in.groupby('txId')['input_address'].nunique().reset_index()
        fan_in.columns = ['txId', 'fan_in']

        df = df.merge(fan_out, on='txId', how='left')
        df = df.merge(fan_in, on='txId', how='left')
        df[['fan_in', 'fan_out']] = df[['fan_in', 'fan_out']].fillna(0).astype(int)

        # Fraud indicators
        df['many_outputs_flag'] = (df['fan_out'] >= 3).astype(int)

        positive_out_mean = df[df['out_BTC_mean'] > 0]['out_BTC_mean']
        if len(positive_out_mean) > 0:
            small_threshold = positive_out_mean.quantile(0.20)
            df['small_outputs_flag'] = (df['out_BTC_mean'] <= small_threshold).astype(int)
        else:
            df['small_outputs_flag'] = 0

        df['out_range_ratio'] = (
            (df['out_BTC_max'] - df['out_BTC_min']) /
            (df['out_BTC_mean'] + 1e-9)
        )
        df['equal_split_flag'] = (df['out_range_ratio'] < 0.1).astype(int)

        df['peel_chain_flag'] = (
            (df['fan_in'] == 1) &
            (df['fan_out'] == 2)
        ).astype(int)

        # Network features
        df['in_out_ratio'] = df['in_degree'] / (df['out_degree'] + 1e-9)
        df['degree_sum'] = df['in_degree'] + df['out_degree']
        df['degree_diff'] = df['in_degree'] - df['out_degree']

        # BTC flow
        df['btc_flow_ratio'] = df['in_BTC_total'] / (df['out_BTC_total'] + 1e-9)
        df['btc_concentration'] = df['out_BTC_max'] / (df['out_BTC_total'] + 1e-9)

        df['tx_complexity'] = (
            df['num_input_addresses'] * df['num_output_addresses']
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        df[numeric_cols] = df[numeric_cols].fillna(0)

        engineered_features = [
            'fan_in', 'fan_out', 'many_outputs_flag', 'small_outputs_flag',
            'equal_split_flag', 'peel_chain_flag', 'in_out_ratio',
            'degree_sum', 'degree_diff', 'btc_flow_ratio',
            'btc_concentration', 'tx_complexity'
        ]

        print(f"Added {len(engineered_features)} features")

        labeled_df = df[df['class'].isin([1, 2])]
        if len(labeled_df) > 0:
            illicit_df = labeled_df[labeled_df['class'] == 1]
            licit_df = labeled_df[labeled_df['class'] == 2]

            print(f"Illicit transactions with equal splits: {illicit_df['equal_split_flag'].mean():.1%}")
            print(f"Licit transactions with equal splits: {licit_df['equal_split_flag'].mean():.1%}")

        return df, engineered_features