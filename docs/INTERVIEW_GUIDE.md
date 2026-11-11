# GraphShield: Technical Interview & Discussion Guide

This guide is designed for preparing machine learning and security engineering interviews. It uses the STAR framework (Situation, Task, Action, Result) and covers the high-yield technical questions interviewers ask.

---

## 1. The 60-Second Elevator Pitch

> *"I developed **GraphShield**, a secure real-time fraud detection platform using **Inductive Graph Neural Networks (GraphSAGE)**. In traditional banking architectures, tabular models evaluate transactions as isolated rows, which misses collusive fraud rings where multiple compromised cards interact with shared merchant networks.
> 
> To solve this, I modeled transactions in a heterogeneous bipartite graph linking cards, merchants, and transactions. I addressed the real-time latency challenge by implementing a **bounded 2-hop neighborhood sampling algorithm** that computes inductive fraud probabilities in under 5 milliseconds without rebuilding the underlying graph.
> 
> Because financial ML requires strict regulatory adherence, I integrated defense-in-depth security: salted HMAC-SHA256 tokenization to protect cardholder PII, role-based masking for analysts, and a cryptographic HMAC-chained audit log that aligns with PCI-DSS and GDPR requirements."*

---

## 2. Core Technical Questions & Confident Answers

### Q1: Why use a Graph Neural Network instead of XGBoost or LightGBM?
**Answer:**
> "Tabular models treat each transaction as an independent, identically distributed ($i.i.d.$) row. However, credit card fraud is inherently **relational and structural**. Fraudsters operate through:
> 1. Compromised merchant terminals leaking customer data.
> 2. Card cycling attacks across rapid merchant hops.
> 3. Synthetic identity rings transacting with colluding shell merchants.
> 
> A GNN propagates behavioral embeddings across the graph structure. If a merchant is connected to confirmed fraudulent transactions, neighborhood aggregation passes that risk signal into every subsequent transaction at that merchant, catching fraud patterns invisible to isolated row models."

### Q2: How does GraphSAGE handle unseen cards in real-time?
**Answer:**
> "Transductive GNNs (like vanilla GCN) learn a fixed lookup embedding per node ID and require the full adjacency matrix, meaning an unseen card requires re-training or re-indexing the entire graph.
> 
> In contrast, **GraphSAGE is inductive**: it learns **generalized aggregation functions** (shared linear projections and pooling layers) over local node feature distributions. When an unseen card or merchant transacts for the first time, GraphSAGE samples its immediate neighbors and computes the embedding on the fly."

### Q3: How did you handle the extreme 0.12% class imbalance?
**Answer:**
> "In the 24.39M Kaggle dataset, only ~0.12% of transactions are fraudulent (1 fraud per 820 legitimate transactions). Naive Cross-Entropy causes the gradient to be dominated by the majority class, leading to high accuracy ($99.88\%$) but zero recall.
> 
> We tackled this with two distinct loss formulations:
> 1. **Class-Weighted Cross-Entropy:** Weighting positive fraud loss by $w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}} \approx 820$.
> 2. **Focal Loss ($\gamma = 2.0, \alpha = 0.25$):** Dynamically down-weighting the loss contribution of easy-to-classify legitimate transactions and focusing gradient updates on hard borderline anomalies.
> 3. **Metric Selection:** We ban Accuracy and judge exclusively on **Precision-Recall AUC (PR-AUC)**, Precision, Recall, and F1-Score."

### Q4: How is security integrated into the machine learning pipeline?
**Answer:**
> "We apply defense-in-depth principles across the entire ML lifecycle:
> - **Data Privacy (PCI-DSS & GDPR):** Raw Primary Account Numbers (PANs) and User IDs are pseudonymized with salted HMAC-SHA256 before graph construction. This prevents precomputed Rainbow Table attacks if graph storage is ever breached.
> - **Tamper-Evident Audit Trails:** Predictions and alert configurations are logged to an append-only JSONL file where each entry contains an HMAC-SHA256 signature chained to the previous entry's hash. Any tampering or deletion breaks the cryptographic chain.
> - **Model Integrity:** We store a SHA-256 checksum of the trained `.pt` weights and verify it before memory loading, preventing unauthorized model tampering or backdoor injection."

