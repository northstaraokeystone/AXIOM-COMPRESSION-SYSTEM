"""Economy module - Receipt-gated resource allocation.

Resource management with cryptographic proof of authorization.
"""

from .receipt_economy import (
    authorize_resource,
    check_authorization,
    revoke_authorization,
    emit_authorization_receipt,
    authorize_with_receipt,
    verify_receipt_for_access,
    emit_access_receipt,
    record_operation_cost,
    ResourceAuthorization,
)

from .cost_accounting import (
    track_operation_cost,
    get_cost_summary,
    allocate_budget,
    emit_cost_receipt,
    OperationCost,
    CostBudget,
)

from .quota_enforcement import (
    check_quota,
    consume_quota,
    reset_quota,
    emit_quota_receipt,
    QuotaStatus,
    QuotaConfig,
)

__all__ = [
    # Receipt economy
    "authorize_resource",
    "check_authorization",
    "revoke_authorization",
    "emit_authorization_receipt",
    "authorize_with_receipt",
    "verify_receipt_for_access",
    "emit_access_receipt",
    "record_operation_cost",
    "ResourceAuthorization",
    # Cost accounting
    "track_operation_cost",
    "get_cost_summary",
    "allocate_budget",
    "emit_cost_receipt",
    "OperationCost",
    "CostBudget",
    # Quota enforcement
    "check_quota",
    "consume_quota",
    "reset_quota",
    "emit_quota_receipt",
    "QuotaStatus",
    "QuotaConfig",
]
