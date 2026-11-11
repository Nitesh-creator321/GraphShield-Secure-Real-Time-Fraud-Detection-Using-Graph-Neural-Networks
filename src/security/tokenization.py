"""
Data Privacy, Salted Tokenization & Pseudonymization Module for GraphShield.

Implements Requirement 8.B:
- Salted Cryptographic Tokenization of Sensitive Identifiers (Cards, Users, Merchants)
- Rainbow-table resistant HMAC-SHA256 pseudonymization
- Display masking for compliance with PCI-DSS Requirement 3 and GDPR / India DPDP Act
"""

import hmac
import hashlib
from typing import Optional
from src.security.security_config import sec_config


def salted_hash_token(raw_value: str, prefix: str = "tok") -> str:
    """
    Transforms a raw identifier (Card number, User ID, Merchant ID)
    into a cryptographically salted, non-reversible pseudonym token.

    Why Salted HMAC-SHA256?
    - Standard MD5 or SHA-256 without a salt can be reversed using precomputed
      Rainbow Tables (dictionaries of millions of known card numbers).
    - Salted HMAC uses a private secret key known only to the bank server,
      making brute-force reversal mathematically infeasible.

    Example:
        raw: "card_452_0" -> "tok_c_9f81a7b2"
    """
    if not raw_value:
        return f"{prefix}_unknown"

    raw_bytes = str(raw_value).strip().encode("utf-8")
    token_hash = hmac.new(sec_config.SALT_KEY, raw_bytes, hashlib.sha256).hexdigest()
    # Use first 12 characters of hash for compact representation
    return f"{prefix}_{token_hash[:12]}"


def mask_card_display(card_str: str) -> str:
    """
    Masks a card identifier for UI and analyst display.
    Example:
        "card_452_0" -> "card_***_0"
        "1234-5678-9876-5432" -> "****-****-****-5432"
    """
    card_str = str(card_str).strip()
    if "_" in card_str:
        parts = card_str.split("_")
        if len(parts) >= 3:
            return f"card_***_{parts[-1]}"
        return "card_***"

    if len(card_str) >= 4:
        return f"****-****-****-{card_str[-4:]}"

    return "card_***"


def mask_merchant_display(merch_str: str) -> str:
    """
    Masks a merchant terminal ID for UI display.
    Example:
        "merchant_3527213279" -> "merch_***3279"
    """
    merch_str = str(merch_str).strip()
    if "_" in merch_str:
        parts = merch_str.split("_")
        terminal_id = parts[-1]
        if len(terminal_id) >= 4:
            return f"merch_***{terminal_id[-4:]}"
        return "merch_***"

    if len(merch_str) >= 4:
        return f"merch_***{merch_str[-4:]}"

    return "merch_***"

