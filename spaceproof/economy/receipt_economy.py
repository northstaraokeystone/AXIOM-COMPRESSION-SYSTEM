"""receipt_economy.py - Receipt-gated resource allocation.

Resources require valid receipt authorization for access.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from spaceproof.core import dual_hash, emit_receipt

# === CONSTANTS ===

ECONOMY_TENANT = "spaceproof-economy"


@dataclass
class ResourceAuthorization:
    """Authorization for resource access."""

    auth_id: str
    resource_id: str
    resource_type: str
    grantee_id: str
    permissions: List[str]
    expires_at: str
    granted_by: str
    authorization_hash: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    revoked: bool = False
    revoked_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "auth_id": self.auth_id,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "grantee_id": self.grantee_id,
            "permissions": self.permissions,
            "expires_at": self.expires_at,
            "granted_by": self.granted_by,
            "authorization_hash": self.authorization_hash,
            "created_at": self.created_at,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
        }

    def is_valid(self) -> bool:
        """Check if authorization is still valid."""
        if self.revoked:
            return False
        now = datetime.utcnow().isoformat() + "Z"
        return now < self.expires_at


# Authorization storage
_authorizations: Dict[str, ResourceAuthorization] = {}


def authorize_resource(
    resource_id: str,
    resource_type: str,
    grantee_id: str,
    permissions: List[str],
    granted_by: str,
    duration_hours: int = 24,
) -> ResourceAuthorization:
    """Create resource authorization.

    Args:
        resource_id: Resource identifier
        resource_type: Type of resource
        grantee_id: Who receives authorization
        permissions: List of permissions (read, write, execute, etc.)
        granted_by: Who grants authorization
        duration_hours: Authorization duration

    Returns:
        ResourceAuthorization
    """
    auth_id = str(uuid.uuid4())
    expires_at = (datetime.utcnow() + timedelta(hours=duration_hours)).isoformat() + "Z"

    # Create authorization hash
    auth_data = {
        "auth_id": auth_id,
        "resource_id": resource_id,
        "grantee_id": grantee_id,
        "permissions": sorted(permissions),
        "expires_at": expires_at,
    }
    auth_hash = dual_hash(str(auth_data))

    auth = ResourceAuthorization(
        auth_id=auth_id,
        resource_id=resource_id,
        resource_type=resource_type,
        grantee_id=grantee_id,
        permissions=permissions,
        expires_at=expires_at,
        granted_by=granted_by,
        authorization_hash=auth_hash,
    )

    _authorizations[auth_id] = auth

    return auth


def check_authorization(
    resource_id: str,
    grantee_id: str,
    required_permission: str,
) -> Optional[ResourceAuthorization]:
    """Check if grantee has permission for resource.

    Args:
        resource_id: Resource identifier
        grantee_id: Grantee to check
        required_permission: Permission needed

    Returns:
        Valid ResourceAuthorization or None
    """
    for auth in _authorizations.values():
        if auth.resource_id == resource_id and auth.grantee_id == grantee_id:
            if auth.is_valid() and required_permission in auth.permissions:
                return auth

    return None


def revoke_authorization(
    auth_id: str,
    revoked_by: str,
    reason: str = "",
) -> bool:
    """Revoke an authorization.

    Args:
        auth_id: Authorization ID
        revoked_by: Who revokes
        reason: Revocation reason

    Returns:
        True if revoked successfully
    """
    auth = _authorizations.get(auth_id)
    if not auth:
        return False

    auth.revoked = True
    auth.revoked_at = datetime.utcnow().isoformat() + "Z"

    # Emit revocation receipt
    emit_receipt(
        "authorization_revoked",
        {
            "tenant_id": ECONOMY_TENANT,
            "auth_id": auth_id,
            "resource_id": auth.resource_id,
            "revoked_by": revoked_by,
            "reason": reason,
        },
    )

    return True


def emit_authorization_receipt(auth: ResourceAuthorization) -> Dict[str, Any]:
    """Emit authorization receipt.

    Args:
        auth: ResourceAuthorization to emit

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "resource_allocation",
        {
            "tenant_id": ECONOMY_TENANT,
            **auth.to_dict(),
        },
    )


def get_authorizations_for_grantee(grantee_id: str) -> List[ResourceAuthorization]:
    """Get all authorizations for a grantee.

    Args:
        grantee_id: Grantee identifier

    Returns:
        List of ResourceAuthorization objects
    """
    return [a for a in _authorizations.values() if a.grantee_id == grantee_id and a.is_valid()]


def clear_authorizations() -> None:
    """Clear all authorizations (for testing)."""
    global _authorizations
    _authorizations = {}


# Aliases for API compatibility
def authorize_with_receipt(
    resource_id: str,
    resource_type: str,
    grantee_id: str,
    permissions: List[str],
    granted_by: str,
    duration_hours: int = 24,
) -> Dict[str, Any]:
    """Authorize resource access and emit receipt.

    Args:
        resource_id: Resource identifier
        resource_type: Type of resource
        grantee_id: Who receives authorization
        permissions: List of permissions
        granted_by: Who grants authorization
        duration_hours: Authorization duration

    Returns:
        Authorization dict with receipt
    """
    auth = authorize_resource(
        resource_id, resource_type, grantee_id, permissions, granted_by, duration_hours
    )
    receipt = emit_authorization_receipt(auth)
    return {
        "authorized": True,
        "authorization": auth.to_dict(),
        "receipt": receipt,
    }


def verify_receipt_for_access(
    receipt: Dict[str, Any],
    resource_id: str,
    required_permission: str,
) -> bool:
    """Verify receipt grants access to resource.

    Args:
        receipt: Receipt to verify
        resource_id: Resource being accessed
        required_permission: Permission needed

    Returns:
        True if receipt grants access
    """
    if not receipt:
        return False

    # Check receipt contains authorization info
    payload = receipt.get("payload", {})
    if payload.get("resource_id") != resource_id:
        return False

    permissions = payload.get("permissions", [])
    return required_permission in permissions


def emit_access_receipt(
    resource_id: str,
    grantee_id: str,
    permission_used: str,
    action: str,
) -> Dict[str, Any]:
    """Emit receipt for resource access.

    Args:
        resource_id: Resource accessed
        grantee_id: Who accessed
        permission_used: Permission used
        action: Action performed

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "resource_access",
        {
            "tenant_id": ECONOMY_TENANT,
            "resource_id": resource_id,
            "grantee_id": grantee_id,
            "permission_used": permission_used,
            "action": action,
        },
    )


def record_operation_cost(
    operation_id: str,
    operation_type: str,
    cost: float,
    unit: str = "credits",
) -> Dict[str, Any]:
    """Record cost of an operation (alias for track_operation_cost).

    Args:
        operation_id: Operation identifier
        operation_type: Type of operation
        cost: Cost amount
        unit: Cost unit

    Returns:
        Cost record dict
    """
    return {
        "operation_id": operation_id,
        "operation_type": operation_type,
        "cost": cost,
        "unit": unit,
        "recorded_at": datetime.utcnow().isoformat() + "Z",
    }
