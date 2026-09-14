"""Explicit confidence policy: there is no default production threshold."""

import math
from dataclasses import dataclass


def probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, not bool")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return float(value)


@dataclass(frozen=True)
class VerificationPolicy:
    confidence_threshold: float
    policy_version: str = "confidence_gate_v1"

    def __post_init__(self):
        probability(self.confidence_threshold, "confidence_threshold")
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string")


@dataclass(frozen=True)
class AutoVerificationAuditConfig:
    random_audit_fraction: float = 0.0
    seed: int = 0

    def __post_init__(self):
        probability(self.random_audit_fraction, "random_audit_fraction")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("audit seed must be an integer")
