"""Tests for spaceproof.economy module."""

from spaceproof.economy import (
    authorize_resource,
    check_authorization,
    ResourceAuthorization,
    track_operation_cost,
    get_cost_summary,
    allocate_budget,
    OperationCost,
    CostBudget,
    check_quota,
    consume_quota,
    reset_quota,
    emit_quota_receipt,
    QuotaStatus,
)


def test_authorize_resource():
    """authorize_resource creates valid authorization."""
    auth = authorize_resource(
        resource_id="compute-001",
        resource_type="compute",
        grantee_id="user-123",
        permissions=["read", "write"],
        granted_by="admin-001",
    )
    assert isinstance(auth, ResourceAuthorization)
    assert auth.resource_id == "compute-001"
    assert auth.grantee_id == "user-123"
    assert "read" in auth.permissions


def test_check_authorization_valid():
    """check_authorization finds valid authorization."""
    authorize_resource(
        resource_id="storage-001",
        resource_type="storage",
        grantee_id="user-check",
        permissions=["read"],
        granted_by="admin-001",
    )
    result = check_authorization(
        resource_id="storage-001",
        grantee_id="user-check",
        required_permission="read",
    )
    assert result is not None
    assert result.grantee_id == "user-check"


def test_check_authorization_no_permission():
    """check_authorization returns None when no permission."""
    authorize_resource(
        resource_id="storage-002",
        resource_type="storage",
        grantee_id="user-limited",
        permissions=["read"],
        granted_by="admin-001",
    )
    result = check_authorization(
        resource_id="storage-002",
        grantee_id="user-limited",
        required_permission="write",
    )
    assert result is None


def test_track_operation_cost():
    """track_operation_cost records cost."""
    cost = track_operation_cost(
        operation_type="inference",
        resource_units=10.5,
        actor_id="user-123",
    )
    assert isinstance(cost, OperationCost)
    assert cost.resource_units == 10.5
    assert cost.total_cost > 0


def test_get_cost_summary():
    """get_cost_summary returns cost aggregation."""
    # Track some costs
    track_operation_cost(
        operation_type="compute",
        resource_units=5.0,
        actor_id="user-summary",
    )
    summary = get_cost_summary(actor_id="user-summary")
    assert isinstance(summary, dict)
    assert "total_cost" in summary


def test_allocate_budget():
    """allocate_budget creates budget."""
    budget = allocate_budget(owner_id="user-budget", amount=1000.0)
    assert isinstance(budget, CostBudget)
    assert budget.remaining_amount == 1000.0


def test_check_quota():
    """check_quota returns quota status."""
    from spaceproof.economy.quota_enforcement import create_quota

    create_quota(
        resource_type="api_calls",
        limit=1000,
        actor_id="user-quota",
    )
    status = check_quota(
        actor_id="user-quota",
        resource_type="api_calls",
    )
    assert isinstance(status, QuotaStatus)
    assert status.limit == 1000


def test_consume_quota():
    """consume_quota deducts from quota."""
    from spaceproof.economy.quota_enforcement import create_quota

    create_quota(
        resource_type="requests",
        limit=100,
        actor_id="user-consume",
    )
    success, status = consume_quota(
        actor_id="user-consume",
        resource_type="requests",
        amount=1,
    )
    assert success is True
    assert status.used >= 1


def test_reset_quota():
    """reset_quota resets usage."""
    from spaceproof.economy.quota_enforcement import create_quota

    create_quota(
        resource_type="resets",
        limit=100,
        actor_id="user-reset",
    )
    consume_quota(actor_id="user-reset", resource_type="resets", amount=10)
    result = reset_quota(actor_id="user-reset", resource_type="resets")
    assert result is True


def test_emit_quota_receipt():
    """emit_quota_receipt emits valid receipt."""
    status = QuotaStatus(
        quota_id="test-quota",
        resource_type="api_calls",
        limit=1000,
        used=50,
        remaining=950,
        window_start="2024-01-01T00:00:00Z",
        window_end="2024-01-02T00:00:00Z",
        is_exceeded=False,
        utilization_pct=5.0,
    )
    receipt = emit_quota_receipt(status)
    assert receipt["receipt_type"] == "quota_enforcement"
