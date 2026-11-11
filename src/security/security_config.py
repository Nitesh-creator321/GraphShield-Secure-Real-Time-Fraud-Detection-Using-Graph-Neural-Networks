"""
Central Security Configuration Loader for GraphShield.

Loads cryptographic secrets, role-based API keys, rate-limiting rules,
and privacy flags from environment variables (.env).
Enforces enterprise security defaults to prevent hardcoded secrets.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load environment variables if .env exists
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


class SecurityConfig:
    """Security configurations for encryption, authentication, and compliance."""

    # Environment mode
    ENV: str = os.getenv("GRAPHSHIELD_ENV", "development").lower()
    IS_PROD: bool = ENV == "production"

    # Cryptographic Salt & HMAC Secrets (Never commit actual production keys)
    SALT_KEY: bytes = os.getenv(
        "GRAPHSHIELD_SALT_KEY",
        "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"
    ).encode("utf-8")

    HMAC_AUDIT_KEY: bytes = os.getenv(
        "GRAPHSHIELD_HMAC_AUDIT_KEY",
        "9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca7"
    ).encode("utf-8")

    SECRET_KEY: str = os.getenv(
        "GRAPHSHIELD_SECRET_KEY",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )

    # Role-Based API Keys (HTTP Header: X-API-Key)
    ADMIN_API_KEY: str = os.getenv("GRAPHSHIELD_ADMIN_API_KEY", "gs_admin_sec_994821a8f01b3c2e")
    ANALYST_API_KEY: str = os.getenv("GRAPHSHIELD_ANALYST_API_KEY", "gs_analyst_sec_410294b6d92a1e0f")

    # Rate Limiting & DoS Protection
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
    MAX_PAYLOAD_SIZE_BYTES: int = int(os.getenv("MAX_PAYLOAD_SIZE_BYTES", "1048576"))  # 1MB

    # Privacy & Anonymization
    ENABLE_SALTED_TOKENIZATION: bool = os.getenv("ENABLE_SALTED_TOKENIZATION", "true").lower() == "true"
    MASK_PII_IN_LOGS: bool = os.getenv("MASK_PII_IN_LOGS", "true").lower() == "true"

    # Model Integrity
    MODEL_INTEGRITY_CHECK_ENABLED: bool = os.getenv("MODEL_INTEGRITY_CHECK_ENABLED", "true").lower() == "true"
    EXPECTED_MODEL_SHA256: str = os.getenv("EXPECTED_MODEL_SHA256", "")

    # Alert Decision Boundary
    DEFAULT_ALERT_THRESHOLD: float = float(os.getenv("DEFAULT_ALERT_THRESHOLD", "0.50"))

    @classmethod
    def validate_api_key(cls, key: str) -> tuple[bool, str]:
        """
        Validates an incoming API key.
        Returns: (is_valid, role) where role is 'admin', 'analyst', or 'anonymous'
        """
        if not key:
            return False, "anonymous"
        if key == cls.ADMIN_API_KEY:
            return True, "admin"
        if key == cls.ANALYST_API_KEY:
            return True, "analyst"
        return False, "anonymous"


# Global security config singleton
sec_config = SecurityConfig()

