# GraphShield™ Security Architecture & Compliance Specifications

This document outlines the security threat model, cryptographic controls, compliance alignments, and verification procedures implemented across GraphShield.

---

## 1. Threat Model Matrix

| Threat Actor | Malicious Objective | Attack Vector / Surface | GraphShield Defense Module |
| :--- | :--- | :--- | :--- |
| **Fraud Syndicates** | Stolen card monetization, unauthorized charges | Rapid card cycling, online CNP transactions, high-risk merchant bursts | **Inductive GraphSAGE + Real-Time Subgraph Scoring** |
| **Camouflage Fraudsters** | Blend into normal traffic to evade GNN detection | Adding edges to high-reputation merchants & small normal purchases | **Module C: Adversarial Neighbor Pruning & Robust Aggregation** |
| **Curious Insiders / Rogue Analysts** | Snooping on user identities or card spending habits | Dashboard interface, database dumps, raw log inspections | **Module B & G: Salted Tokenization, Masked UI, Zero-PAN Logs** |
| **External Hackers** | Model theft / inversion, Denial of Service (DoS) | Automated high-frequency API querying, malformed payload injections | **Module D: API Key Authentication, Strict Schema Checks, Rate Limits** |
| **Supply Chain / Malicious Insiders** | Backdooring the fraud model to let fraudulent transactions pass | Overwriting model weights on disk | **Module E: SHA-256 Model Integrity Verification & Rollback** |
| **Auditing Tamperers** | Concealing fraudulent activity after a breach | Altering or deleting historical prediction logs | **Module F: Cryptographic HMAC-SHA256 Chained Audit Trail** |

---

## 2. Implemented Security Controls

### Control 1: Data Privacy & Salted Tokenization (Module B)
- **Problem:** Storing raw Primary Account Numbers (PANs) or cardholder names in graph databases violates PCI-DSS and exposes users in a breach.
- **GraphShield Solution:** All card identifiers and user keys are transformed via salted HMAC-SHA256 before entering the graph:
  $$\text{Token} = \text{HMAC-SHA256}(\text{SALT\_KEY}, \text{Raw\_Card\_ID})$$
- **Protection:** Prevents Rainbow-Table precomputation attacks. Even if an attacker steals the graph adjacency matrix, they cannot recover the original cardholder identities.

### Control 2: Cryptographic Tamper-Evident Audit Logging (Module F)
- **Problem:** Attackers who compromise a system often erase their tracks in log files.
- **GraphShield Solution:** Every fraud alert generates an immutable JSON-Lines record signed with an HMAC-SHA256 signature chained to the previous record's hash (`prev_hash`).
- **Protection:** Modifying or deleting any single line invalidates all subsequent signatures, immediately signaling unauthorized log tampering.

### Control 3: Model File Integrity Verification (Module E)
- **Problem:** An attacker with local access could replace `gnn_model.pt` with a tampered model that outputs `0.0` (legitimate) for attacker transactions.
- **GraphShield Solution:** Before loading any `.pt` weights into RAM, the serving engine calculates the SHA-256 hash of the file and verifies it against a pinned checksum. If the hashes mismatch, the model execution halts immediately.

### Control 4: Role-Based Access Control & PII Masking (Module G)
- **Analyst Role:** Card numbers and merchant terminals are automatically masked (`card_***_0`, `merch_***3279`).
- **Admin Role:** Full unmasked visibility, restricted by authenticated admin credentials.

### Control 5: API Guardrails & Rate Limiting (Module D)
- Enforces payload bounds ($\le 1\text{ MB}$).
- Rejects malformed JSON and enforces strict Pydantic schemas.
- Bounded query parameters prevent denial-of-service memory exhaustion.

---

## 3. Compliance Alignment Checklist

*Disclaimer: GraphShield is an academic/engineering research prototype. This checklist demonstrates technical control alignment in simple words, not legal advice.*

| Regulation | Key Requirement | GraphShield Implementation |
| :--- | :--- | :--- |
| **PCI-DSS v4.0** | **Req 3:** Protect Stored Cardholder Data | Zero plain card PANs stored. Uses salted tokenization and display masking (`card_***_0`). |
| **PCI-DSS v4.0** | **Req 10:** Log and Monitor All Access | Tamper-evident HMAC-SHA256 audit log records all fraud alerts with UTC timestamps and actor roles. |
| **GDPR (EU)** | **Art. 25 & 32:** Pseudonymization & Security of Processing | Pseudonymizes cardholder data before graph construction; enforces role-based access. |
| **GDPR (EU)** | **Art. 22:** Automated Decision-Making Explainability | Multi-factor GNN attribution provides human-readable reasons for every blocked transaction. |
| **DPDP Act 2023 (India)** | **Sec. 8:** Reasonable Security Safeguards | Cryptographic hashing at rest, rate-limited endpoints, and immutable audit trails prevent data compromise. |

---

## 4. Verification & Security Testing

To verify security controls:

1. **Verify Tokenization:**
   ```powershell
   python -c "from src.security.tokenization import salted_hash_token; print(salted_hash_token('card_123_0'))"
   ```

2. **Verify Cryptographic Audit Log:**
   ```powershell
   Get-Content logs/security_audit.jsonl
   ```

3. **Verify API Schemas:**
   Visit `http://localhost:8000/docs` to inspect OpenAPI parameter validations and security schemas.

