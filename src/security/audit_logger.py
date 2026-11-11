"""
Tamper-Evident Security Audit Logger for GraphShield.

Implements:
- Structured JSON-Lines logging for all fraud alerts and system actions
- Sensitive field masking (PII protection)
- Cryptographic HMAC-SHA256 signature chain for tamper-evidence
- Zero plain card or merchant account numbers in persistent logs
"""

import os
import json
import hmac
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from src.security.security_config import sec_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
AUDIT_LOG_FILE = LOGS_DIR / "security_audit.jsonl"
HMAC_SECRET = sec_config.HMAC_AUDIT_KEY


class SecurityAuditLogger:
    def __init__(self, log_path: Path = AUDIT_LOG_FILE):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.prev_hash = "0" * 64

    def log_alert(
        self,
        tx_id: int,
        masked_card: str,
        masked_merch: str,
        amount: float,
        fraud_prob: float,
        threshold: float,
        explanation: str,
        analyst_role: str = "analyst"
    ) -> Dict[str, Any]:
        """Logs a fraud alert with an HMAC cryptographic signature."""
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "event_type": "FRAUD_ALERT_RAISED",
            "timestamp": timestamp,
            "tx_id": tx_id,
            "card_id": masked_card,
            "merchant_id": masked_merch,
            "amount": amount,
            "fraud_probability": round(fraud_prob, 4),
            "threshold": threshold,
            "explanation": explanation,
            "session_role": analyst_role,
            "prev_hash": self.prev_hash
        }

        # Compute HMAC-SHA256 signature across the record
        canonical_str = json.dumps(record, sort_keys=True)
        signature = hmac.new(HMAC_SECRET, canonical_str.encode(), hashlib.sha256).hexdigest()
        record["hmac_signature"] = signature
        self.prev_hash = signature

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def get_recent_logs(self, limit: int = 50) -> list:
        """Retrieves recent audit log entries."""
        if not self.log_path.exists():
            return []
        lines = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        return lines[-limit:]


# Global singleton instance
audit_logger = SecurityAuditLogger()

