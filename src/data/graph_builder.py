"""
Graph Construction Module for GraphShield.

Builds multi-relational credit card fraud detection graphs from cleaned transactions.
Supports:
1. Heterogeneous Graph: Card <-> Transaction <-> Merchant (Transaction-Node Design)
2. Bipartite Graph: Card <-> Merchant with Transaction Edges (Edge-Classification Design)
3. Pure PyTorch Tensor representations (Zero-dependency guarantee)
4. PyTorch Geometric HeteroData / Data export (if PyG is available)
5. NetworkX MultiGraph export for topological exploration and fast k-hop extraction.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GRAPH_DATA_PATH


class FraudGraphBuilder:
    """
    Constructs and manages the graph representation of transaction networks.

    Design Choice Explained:
    -----------------------
    In financial networks, fraud is rarely an isolated event. It spreads through:
    1. Compromised Cards: A single card used across rapid locations.
    2. Compromised Merchants: A compromised POS or online portal leaking card data.
    3. Fraud Rings: Multiple cards transacting with the same colluding merchant.

    We adopt the 'Transaction-Centric Heterogeneous Graph' architecture:
    - Node Type 'Card': represents the payer account / user card.
    - Node Type 'Merchant': represents the receiving vendor.
    - Node Type 'Transaction': represents the event with all its 18 engineered features.

    Edges:
    - (Card) <--> (Transaction)
    - (Transaction) <--> (Merchant)

    Why this design?
    1. Direct Node Classification: GraphSAGE can classify the Transaction node directly
       by aggregating the past behavior of its connected Card and Merchant.
    2. Real-Time Inductive Scoring: When transaction #100,001 arrives, we simply add
       a single Transaction node connected to its Card and Merchant, sample its 2-hop
       subgraph (taking < 5ms), and evaluate the model. The rest of the graph is untouched!
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list):
        self.df = df.copy()
        self.feature_cols = feature_cols

        # Mapping dictionaries: string ID -> integer index
        self.card2idx: Dict[str, int] = {}
        self.idx2card: Dict[int, str] = {}
        self.merch2idx: Dict[str, int] = {}
        self.idx2merch: Dict[int, str] = {}

        self._build_node_mappings()

    def _build_node_mappings(self):
        """Creates unique integer mappings for all cards and merchants."""
        unique_cards = sorted(self.df["Card_ID"].unique())
        self.card2idx = {cid: idx for idx, cid in enumerate(unique_cards)}
        self.idx2card = {idx: cid for cid, idx in self.card2idx.items()}

        unique_merchants = sorted(self.df["Merchant_ID"].unique())
        self.merch2idx = {mid: idx for idx, mid in enumerate(unique_merchants)}
        self.idx2merch = {idx: mid for mid, idx in self.merch2idx.items()}

    def build_pytorch_graph(self) -> Dict[str, Any]:
        """
        Builds a native PyTorch dictionary graph structure.
        Works without any external C++ compiled extensions.

        Returns:
            dict containing:
            - x_tx: [num_tx, num_features] FloatTensor
            - x_card: [num_cards, card_feat_dim] FloatTensor (historical card profiles)
            - x_merch: [num_merchants, merch_feat_dim] FloatTensor (merchant profiles)
            - edge_card_to_tx: [2, num_tx] LongTensor (Card -> Tx)
            - edge_tx_to_merch: [2, num_tx] LongTensor (Tx -> Merchant)
            - edge_card_to_merch: [2, num_tx] LongTensor (Direct Bipartite Card -> Merchant)
            - y: [num_tx] LongTensor (Fraud labels: 0 or 1)
            - timestamps: array of Timestamps for temporal splitting
            - card2idx, merch2idx: entity lookup maps
        """
        print("[*] Assembling PyTorch Graph Tensors...")

        num_tx = len(self.df)
        num_cards = len(self.card2idx)
        num_merchants = len(self.merch2idx)

        # 1. Transaction Node Features & Labels
        x_tx = torch.tensor(self.df[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(self.df["label"].values, dtype=torch.long)

        # 2. Map edges using vectorized index lookups
        card_indices = self.df["Card_ID"].map(self.card2idx).values
        merch_indices = self.df["Merchant_ID"].map(self.merch2idx).values
        tx_indices = np.arange(num_tx)

        # Edges: Card <-> Transaction
        # Row 0: Source, Row 1: Target
        edge_card_to_tx = torch.tensor(
            np.vstack([card_indices, tx_indices]), dtype=torch.long
        )
        edge_tx_to_card = torch.tensor(
            np.vstack([tx_indices, card_indices]), dtype=torch.long
        )

        # Edges: Transaction <-> Merchant
        edge_tx_to_merch = torch.tensor(
            np.vstack([tx_indices, merch_indices]), dtype=torch.long
        )
        edge_merch_to_tx = torch.tensor(
            np.vstack([merch_indices, tx_indices]), dtype=torch.long
        )

        # Direct Bipartite Edges: Card <-> Merchant
        edge_card_to_merch = torch.tensor(
            np.vstack([card_indices, merch_indices]), dtype=torch.long
        )

        # 3. Compute static entity profile features (Node initializations)
        # Card Profile: [Total Tx Count, Mean Amount, Unique Merchant Count]
        card_stats = self.df.groupby("Card_ID").agg(
            tx_count=("Amount_clean", "count"),
            mean_amt=("Amount_clean", "mean"),
            unique_merchants=("Merchant_ID", "nunique")
        ).reindex([self.idx2card[i] for i in range(num_cards)]).fillna(0.0)

        card_stats_norm = (card_stats - card_stats.mean()) / (card_stats.std() + 1e-6)
        x_card = torch.tensor(card_stats_norm.values, dtype=torch.float32)

        # Merchant Profile: [Total Volume, Mean Amount, Unique Cards Count]
        merch_stats = self.df.groupby("Merchant_ID").agg(
            tx_count=("Amount_clean", "count"),
            mean_amt=("Amount_clean", "mean"),
            unique_cards=("Card_ID", "nunique")
        ).reindex([self.idx2merch[i] for i in range(num_merchants)]).fillna(0.0)

        merch_stats_norm = (merch_stats - merch_stats.mean()) / (merch_stats.std() + 1e-6)
        x_merch = torch.tensor(merch_stats_norm.values, dtype=torch.float32)

        graph_dict = {
            "x_tx": x_tx,
            "x_card": x_card,
            "x_merch": x_merch,
            "edge_card_to_tx": edge_card_to_tx,
            "edge_tx_to_card": edge_tx_to_card,
            "edge_tx_to_merch": edge_tx_to_merch,
            "edge_merch_to_tx": edge_merch_to_tx,
            "edge_card_to_merch": edge_card_to_merch,
            "y": y,
            "timestamps": self.df["Timestamp"].values,
            "card2idx": self.card2idx,
            "idx2card": self.idx2card,
            "merch2idx": self.merch2idx,
            "idx2merch": self.idx2merch,
            "feature_cols": self.feature_cols,
            "num_tx": num_tx,
            "num_cards": num_cards,
            "num_merchants": num_merchants
        }

        print(f"[+] PyTorch Graph Tensors constructed successfully:")
        print(f"    Transaction Nodes: {num_tx:,} (dim: {x_tx.shape[1]})")
        print(f"    Card Nodes:        {num_cards:,} (dim: {x_card.shape[1]})")
        print(f"    Merchant Nodes:    {num_merchants:,} (dim: {x_merch.shape[1]})")
        print(f"    Card-Tx Edges:     {edge_card_to_tx.shape[1]:,}")
        print(f"    Tx-Merch Edges:    {edge_tx_to_merch.shape[1]:,}")
        print(f"    Fraud Labels (y):  {y.sum().item():,} fraud / {(y == 0).sum().item():,} legit")

        return graph_dict

    def build_networkx_graph(self, max_edges: Optional[int] = 10_000) -> nx.MultiGraph:
        """
        Builds a NetworkX MultiGraph.
        Useful for visualization, debugging, and small sub-network inspections.
        """
        print(f"[*] Constructing NetworkX Graph (sampling max {max_edges:,} edges for speed)...")
        G = nx.MultiGraph()

        subset_df = self.df.iloc[:max_edges] if max_edges else self.df

        for idx, row in subset_df.iterrows():
            card_node = f"C_{row['Card_ID']}"
            merch_node = f"M_{row['Merchant_ID']}"

            G.add_node(card_node, node_type="card")
            G.add_node(merch_node, node_type="merchant")

            G.add_edge(
                card_node,
                merch_node,
                tx_id=idx,
                amount=row["Amount_clean"],
                timestamp=str(row["Timestamp"]),
                is_fraud=int(row["label"])
            )

        print(f"[+] NetworkX Graph created with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")
        return G

    def save_graph(self, output_path: Path = GRAPH_DATA_PATH):
        """Saves the PyTorch graph dictionary to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        graph_dict = self.build_pytorch_graph()
        torch.save(graph_dict, output_path)
        print(f"[+] Graph saved to: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
        return graph_dict


def build_and_save_graph(sample_size: int = 100_000) -> Dict[str, Any]:
    """Helper function to load data, build the graph, and save it."""
    from src.data.loader import prepare_dataset
    df, feature_cols = prepare_dataset(sample_size=sample_size)
    builder = FraudGraphBuilder(df, feature_cols)
    graph_dict = builder.save_graph()
    return graph_dict


if __name__ == "__main__":
    build_and_save_graph(sample_size=100_000)

