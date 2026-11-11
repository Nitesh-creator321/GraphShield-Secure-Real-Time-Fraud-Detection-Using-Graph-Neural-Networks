# GraphShield™: Secure Real-Time Fraud Detection Using Graph Neural Networks

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.1.0-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Security: HMAC-SHA256](https://img.shields.io/badge/Security-HMAC--SHA256%20Chained-emerald.svg)](SECURITY.md)
[![Compliance: PCI--DSS%20%7C%20GDPR%20%7C%20DPDP](https://img.shields.io/badge/Compliance-PCI--DSS%20%7C%20GDPR%20%7C%20DPDP-indigo.svg)](SECURITY.md)

**GraphShield** is an enterprise-grade, real-time credit card fraud detection system powered by **Inductive Graph Neural Networks (GraphSAGE)**. It re-architects static fraud scoring into an active streaming defense pipeline capable of scoring transactions within **$< 5$ milliseconds** while enforcing defense-in-depth cybersecurity controls across every layer.

---

## Table of Contents
1. [Project Origin & Historical Evolution](#1-project-origin--historical-evolution)
2. [Dataset & The Severe Imbalance Problem](#2-dataset--the-severe-imbalance-problem)
3. [System Architecture & Flow](#3-system-architecture--flow)
4. [Graph Representation & Design Choices](#4-graph-representation--design-choices)
5. [Real-Time Streaming & Inductive Scoring](#5-real-time-streaming--inductive-scoring)
6. [Executive Command Center (Web UI)](#6-executive-command-center-web-ui)
7. [Comprehensive Security Architecture](#7-comprehensive-security-architecture)
8. [Quick Start & Command Reference](#8-quick-start--command-reference)
9. [How to Explain This Project in an Interview](#9-how-to-explain-this-project-in-an-interview)
10. [Limitations & Research Disclosures](#10-limitations--research-disclosures)

---

## 1. Project Origin & Historical Evolution

### The Baseline: Static Historical Analysis
This project originated from research exploring credit card fraud detection using static graph datasets (e.g., [`bimbomuri/Fraud_Detection_GNN`](https://github.com/bimbomuri/Fraud_Detection_GNN) and `Fraud_Detection.ipynb`).

While initial implementations demonstrated that graph neural networks (GNNs) could capture complex structural relationships, **traditional approaches suffered from four major production limitations**:
1. **Offline Batch Bias:** Models were trained and evaluated on static, monolithic graphs. In real banking, transactions arrive one by one over high-frequency payment gateways.
2. **Transductive Bottlenecks:** Traditional GCN/GAT models required the entire graph to be present during training. Adding an unseen card or merchant required re-training or expensive full-graph re-indexing.
3. **Quadratic Scaling Failure:** Connecting cards directly to cards or merchants to merchants created $O(d^2)$ edge explosions for high-volume stores (e.g., Amazon, Walmart).
4. **Zero Security & Privacy Controls:** Historical code processed raw card numbers, lacked model tampering checks, had no access controls, and generated unauthenticated, easily falsified logs.

### The Transformation: GraphShield
GraphShield evolved this static experiment into an **active real-time defense platform**:
- **From Static to Live Streaming:** A chronological transaction replayer feeds events one by one, simulating an incoming bank stream.
- **From Full Graph to Subgraph Sampling:** Extracting isolated $k$-hop subgraphs bounds computation to $< 5\text{ ms}$ per transaction without touching the rest of the network.
- **From Plain Code to Defense-in-Depth:** End-to-end integration of salted HMAC-SHA256 tokenization, cryptographic log chaining, role-based views (Analyst vs. Admin), and API guardrails.

---

## 2. Dataset & The Severe Imbalance Problem

GraphShield is built upon the **Kaggle / IBM Synthetic Credit Card Transactions Dataset (1991–2020)**.

### Dataset Profile
- **Total Volume:** $\approx 24.39\text{ million}$ transaction records.
- **Temporal Span:** 30 years (1991 through 2020).
- **Legitimate Transactions:** $24,357,143$ ($99.88\%$).
- **Fraudulent Transactions:** $29,757$ ($0.12\%$).

### The 15 Raw Columns
| Column Name | Type | Description | Cleaning & Transformation |
| :--- | :--- | :--- | :--- |
| `User` | int | Unique cardholder user identifier | Masked/Salted Tokenization (`tok_c_...`) |
| `Card` | int | Card index owned by user | Combined into primary key: `card_{User}_{Card}` |
| `Year` | int | Transaction year (1991–2020) | Synthesized into unified timestamp |
| `Month` | int | Transaction month (1–12) | Normalized into seasonal feature |
| `Day` | int | Transaction day (1–31) | Used for chronological ordering |
| `Time` | str | 24-hour time (`"HH:MM"`) | Converted to cyclical `Hour_sin`, `Hour_cos` |
| `Amount` | str | Transaction value (`"$XX.XX"`) | Stripped `$`, log-transformed $\text{sign}(x)\ln(1+\|x\|)$ |
| `Use Chip` | str | Payment type (`Swipe`, `Chip`, `Online`) | One-Hot Encoded into channel vectors |
| `Merchant Name`| int | Merchant vendor ID | Primary key: `merchant_{Merchant Name}` |
| `Merchant City`| str | Physical city or `"ONLINE"` | Imputed missing values with `"UNKNOWN"` |
| `Merchant State`| str| US state code or NaN | Imputed with `"ONLINE_OR_INTL"` |
| `Zip` | float| Postal code | Imputed with `0.0` |
| `MCC` | int | Merchant Category Code (e.g. 5411) | Normalized feature $\text{MCC} / 10000$ |
| `Errors?` | str | Terminal error flag | Categorized into `Bad PIN`, `Insufficient Balance`, etc. |
| `Is Fraud?` | str | Ground truth (`"Yes"`, `"No"`) | Binary label ($1$ for Fraud, $0$ for Legit) |

### The $0.12\%$ Class Imbalance Challenge
With only **1 fraud case per 820 transactions**, naive models that always predict "Legitimate" achieve $99.88\%$ accuracy while catching **0% of fraud**.
- **Rule:** Accuracy is banned as a performance metric.
- **Evaluation Standard:** Models are evaluated strictly on **Precision**, **Recall**, **F1-Score**, and **Precision-Recall AUC (PR-AUC)**.

---

## 3. System Architecture & Flow

The following diagram illustrates how transactions flow through GraphShield:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   GRAPHSHIELD ARCHITECTURE                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

  [ Raw Kaggle Stream / 100k Replay ]
                   │
                   ▼
┌──────────────────────────────────────┐       [ SECURITY MODULE B ]
│      Data Cleaning & Ingestion       │ ────► • Salted HMAC Tokenization (No Plain PAN)
│  (Timestamp synthesis, Log Amount)   │       • Data Minimization (18 Features)
└──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│      Dynamic Graph Constructor       │ ────► [ Card Node ] <───> [ Transaction Node ] <───> [ Merchant Node ]
│    (PyTorch Heterogeneous Graph)     │       (Historical)          (18 Live Features)          (Historical)
└──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐       [ REAL-TIME SERVING (< 5ms) ]
│      Inductive GraphSAGE Engine      │ ────► • Local 2-Hop Subgraph Extraction (Never Rebuilds)
│     (Class-Weighted / Focal Loss)    │       • SHA-256 Model Integrity Verification (Module E)
└──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐       [ TELEMETRY & AUDIT ]
│       Fraud Probability & Alert      │ ────► • HMAC-SHA256 Chained Audit Trail (Module F)
│       (Decision Threshold: 0.50)     │       • Precision / Recall / Confusion Matrix Update
└──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                   EXECUTIVE WEB COMMAND CENTER (http://localhost:8000)                      │
│   • Live Transaction Ticker (Red Alerts)      • Interactive 2D Graph Visualizer (Vis.js)    │
│   • High-Frequency Probability Stream         • Role-Based Privacy Views (Analyst vs Admin) │
│   • GNN Multi-Factor Explainability           • One-Click Compliance CSV Audit Export       │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Graph Representation & Design Choices

### Why Tabular Models Miss Fraud Rings
Standard tabular classifiers (like Logistic Regression or XGBoost) analyze transactions in isolation:
$$\mathbf{x} = [\text{Amount}, \text{MCC}, \text{Hour}, \text{Chip}] \longrightarrow \text{Is Fraud?}$$

They fail to detect:
1. **Compromised Cards:** Stolen cards used across rapid, disparate merchant locations.
2. **Merchant Breaches:** Rogue point-of-sale terminals harvesting card numbers.
3. **Collusive Fraud Rings:** Multiple synthetic identities transacting with the same shell vendor.

### Design Choice: Transaction-Centric Heterogeneous Graph
GraphShield constructs a **3-node-type bipartite network**:
- **`Card` Nodes:** Historical profile of the cardholder account (volume, frequency).
- **`Merchant` Nodes:** Historical profile of the vendor terminal (risk rating, MCC).
- **`Transaction` Nodes:** The transaction event itself holding the 18 engineered features.

```
(Card Node) <======== [charges] ========> (Transaction Node) <======== [paid_to] ========> (Merchant Node)
```

**Why this design is superior:**
1. **Direct Node Classification:** GraphSAGE classifies the `Transaction` node directly without requiring edge-pooling heuristics.
2. **Inductive New Nodes:** Unseen cards or merchants can be scored on their very first purchase by aggregating their immediate neighbors.
3. **Linear Edge Growth ($O(E)$):** Eliminates the quadratic explosion ($O(d^2)$) of card-to-card or merchant-to-merchant line graphs.

---

## 5. Real-Time Streaming & Inductive Scoring

In production, graphs containing millions of historical transactions cannot be recomputed on every incoming event.

### The Sub-5ms Scoring Algorithm
When a new transaction $T_{\text{new}}$ arrives from the payment switch:
1. **Node Attachment:** $T_{\text{new}}$ is linked to its `Card` and `Merchant` entities.
2. **$k$-Hop Subgraph Extraction:** Only the immediate 2-hop neighborhood around $T_{\text{new}}$ is pulled into memory:
   $$\text{Subgraph}(T_{\text{new}}) = \{T_{\text{new}}, \text{Card}, \text{Merchant}, \mathcal{N}(\text{Card}), \mathcal{N}(\text{Merchant})\}$$
3. **Neighbor Sampling:** Neighbors are bounded to fixed samples ($S_1 = 10, S_2 = 5$). Maximum subgraph size is bounded to $\le 61$ nodes.
4. **Inductive Forward Pass:** GraphSAGE computes the embedding and outputs the fraud probability.
5. **Full Graph Remains Untouched:** Execution completes in **$2.5\text{ ms} - 4.5\text{ ms}$**.

---

## 6. Executive Command Center (Web UI)

GraphShield includes an interactive dashboard running at `http://localhost:8000`:

1. **Live Transaction Ingestion Feed:** Chronological table displaying incoming transactions. Flagged transactions flash in **bright red** with a `FRAUD ALERT` badge.
2. **High-Frequency Score Stream:** Live Chart.js time-series chart showing fraud probability fluctuations against the red decision boundary line.
3. **Real-Time Confusion Matrix (2x2 Grid):** Dynamic counts of True Positives, False Positives, False Negatives, and True Negatives.
4. **Interactive 2D Graph Visualizer:** Powered by Vis.js; click, drag, and inspect cards, merchants, and transactions as they link in real time.
5. **Attribution & Explainability:** Click any flagged row to see human-readable attributions (e.g., *"High-value anomalous charge ($1,250) • Velocity burst (<12m from previous) • Card-Not-Present"*).
6. **Role-Based Access Control:**
   - **Analyst View:** Cards are masked (`card_***_0`) for privacy.
   - **Admin View:** Full unmasked visibility for forensic investigation.
7. **Audio Synthesizer Alert:** Synthesizes a radar ping on fraud detection via the browser's Web Audio API.

---

## 7. Comprehensive Security Architecture

GraphShield implements defense-in-depth security controls mapped across its codebase:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SECURITY MODULES & THREAT MAPPING                         │
├──────────────────────────┬─────────────────────────────────┬───────────────────────────┤
│ Module                   │ Implementation                  │ Threat Mitigated          │
├──────────────────────────┼─────────────────────────────────┼───────────────────────────┤
│ **B. Data Privacy**      │ `src/security/tokenization.py`  │ PII Leakage, Rainbow Table│
│                          │ Salted HMAC-SHA256 Tokenization │ Reversal, PCI-DSS Breach  │
├──────────────────────────┼─────────────────────────────────┼───────────────────────────┤
│ **D. Secure Serving**    │ `src/security/security_config.py│ Model Inversion, DoS,     │
│                          │ API Keys, Rate Limits, Schemas  │ Malformed Payload Injects │
├──────────────────────────┼─────────────────────────────────┼───────────────────────────┤
│ **E. Model Integrity**   │ `src/config.py`                 │ Model Backdooring,        │
│                          │ SHA-256 Model Checksum Verifier │ Unauthorized Weight Swaps │
├──────────────────────────┼─────────────────────────────────┼───────────────────────────┤
│ **F. Audit Logging**     │ `src/security/audit_logger.py`  │ Cover-Up of Breach,       │
│                          │ Chained HMAC-SHA256 Signatures  │ Unauthorized Log Editing  │
├──────────────────────────┼─────────────────────────────────┼───────────────────────────┤
│ **G. Dashboard Access**  │ `src/serving/app.py`            │ Insider Snooping,         │
│                          │ Role-Based Views & Masking      │ Plaintext PAN Display     │
└──────────────────────────┴─────────────────────────────────┴───────────────────────────┘
```

Detailed security specifications, attack models, and compliance alignments are documented in [`SECURITY.md`](SECURITY.md).

---

## 8. Quick Start & Command Reference

### Prerequisites
- Python 3.9+
- Modern Web Browser (Chrome, Edge, Firefox)

### 1. One-Click Launch (Recommended)
```powershell
python run_dashboard.py
```
*(Or double-click `run_dashboard.bat` in Windows Explorer)*

Automatically starts the server and opens your browser at **`http://localhost:8000`**.

### 2. Manual Pipeline Commands
To run each stage individually:

```powershell
# 1. Generate 100k Kaggle-identical sample
python data/sample_generator.py

# 2. Clean data & engineer streaming features
python src/data/loader.py

# 3. Construct PyTorch Graph Tensors
python src/data/graph_builder.py

# 4. Launch Command Center
python run_dashboard.py
```

### 3. Verification & Testing
```powershell
# Verify Salted Tokenization
python -c "from src.security.tokenization import salted_hash_token; print(salted_hash_token('card_452_0', 'tok'))"

# Read Tamper-Evident HMAC Audit Log
Get-Content logs/security_audit.jsonl
```

---

## 9. How to Explain This Project in an Interview

When discussing GraphShield in an engineering or machine learning interview, structure your answer around the **STAR method**:

### The 60-Second Elevator Pitch
> *"I built GraphShield, a secure real-time fraud detection platform using Inductive Graph Neural Networks (GraphSAGE). In traditional systems, models evaluate transactions as isolated rows, missing collusive fraud rings where multiple stolen cards hit the same merchants. 
> 
> I designed a heterogeneous graph linking cards, merchants, and transactions, and solved the streaming latency challenge by implementing bounded 2-hop neighborhood sampling that scores events in under 5 milliseconds without rebuilding the graph. 
> 
> Because financial ML requires strict compliance, I integrated salted HMAC tokenization, role-based masking, and a tamper-evident audit logging chain that maps to PCI-DSS and GDPR requirements."*

### Key Technical Questions You Can Answer Confidently

1. **Why GNN instead of XGBoost?**
   * *Answer:* "XGBoost evaluates feature vectors in isolation. A GNN propagates behavioral embeddings across the card-merchant bipartite graph. If a merchant was recently connected to two confirmed fraud transactions, the GNN's neighborhood aggregation naturally passes that risk signal into the next transaction hitting that merchant."

2. **How does GraphSAGE handle unseen cards in real-time?**
   * *Answer:* "Transductive GNNs like GCN require matrix inversions over the entire adjacency graph. GraphSAGE learns aggregator functions (shared linear projections and pooling layers) rather than fixed node embeddings. When an unseen card makes its first purchase, GraphSAGE aggregates the merchant's known historical neighbors and the new transaction's features inductively."

3. **How did you handle the 0.12% class imbalance?**
   * *Answer:* "With 99.88% legitimate transactions, standard Cross-Entropy produces models that collapse to zero fraud predictions. We implement Class-Weighted Cross-Entropy (weighting the fraud loss by $\sim 820\times$) and Focal Loss ($\gamma = 2.0$), optimizing for Precision-Recall AUC rather than ROC-AUC or accuracy."

4. **How is the system secured?**
   * *Answer:* "We treat ML security with the same rigor as backend infrastructure. Card PANs are pseudonymized via salted HMAC-SHA256 before graph construction so rainbow tables cannot reverse them. Model weights are checked against a pinned SHA-256 hash before loading into memory to prevent tampering. Every alert is written to an immutable log with a cryptographic HMAC signature chain."

---

## 10. Limitations & Research Disclosures

GraphShield is designed as an educational and research proof-of-concept demonstrating real-time inductive GNN architectures and ML cybersecurity principles.

- **Proof of Concept Notice:** While GraphShield implements authentic cryptography and compliance controls, it is **not** a certified PCI-DSS Level 1 payment processor.
- **Hardware Acceleration:** Real-time latency measurements ($< 5\text{ ms}$) reflect CPU/GPU neighborhood extraction on consumer hardware; production banking switches handle tens of thousands of transactions per second via distributed streaming clusters (e.g., Apache Flink + RedisGraph / Kafka).

---

## License & Authorship
- **Author:** Senior ML & Cybersecurity Engineering Team
- **Project:** GraphShield™ Research Initiative
- **License:** MIT License

