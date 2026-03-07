from typing import Optional, Tuple

from fastapi import Depends, HTTPException, Request
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from models import User, get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(
    db: Session, email: str, password: str
) -> Tuple[Optional[User], str]:
    """
    Authenticate a user by email and password.

    Returns (user, reason) where reason is one of:
        "active"              — credentials correct, account active
        "invalid_credentials" — wrong email or password
        "pending"             — account awaiting approval
        "blocked"             — account blocked
    """
    user = db.query(User).filter(User.email == email.strip().lower()).first()
    if not user or not verify_password(password, user.password_hash):
        return None, "invalid_credentials"
    return user, user.status


# ── Custom exceptions ─────────────────────────────────────────────────────────

class RequiresLoginException(Exception):
    pass


class RequiresPasswordChangeException(Exception):
    pass


# ── FastAPI dependencies ───────────────────────────────────────────────────────

def get_session_user(request: Request, db: Session = Depends(get_db)) -> User:
    """
    Returns the logged-in user from session without any extra checks.
    Used only for routes that must be accessible even when must_change_password=True.
    """
    user_id = request.session.get("user_id")
    if not user_id:
        raise RequiresLoginException()
    user = db.query(User).filter(User.id == user_id, User.status == "active").first()
    if not user:
        raise RequiresLoginException()
    return user


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """
    Returns the *effective* user for the request.
    If a superadmin is impersonating another user, returns the impersonated user.
    Raises RequiresLoginException if not authenticated or account not active.
    Raises RequiresPasswordChangeException if the user must change their password first.
    """
    user_id = request.session.get("user_id")
    if not user_id:
        raise RequiresLoginException()

    impersonating_id = request.session.get("impersonating_user_id")
    user = None

    if impersonating_id:
        user = db.query(User).filter(User.id == impersonating_id, User.status == "active").first()
        if not user:
            request.session.pop("impersonating_user_id", None)
            impersonating_id = None

    if not impersonating_id:
        user = db.query(User).filter(User.id == user_id, User.status == "active").first()

    if not user:
        raise RequiresLoginException()

    if user.must_change_password:
        raise RequiresPasswordChangeException()

    return user


def get_actual_user(request: Request, db: Session = Depends(get_db)) -> User:
    """
    Returns the *actual* logged-in user, ignoring any active impersonation.
    Use this for permission checks on admin-only routes.
    Raises RequiresPasswordChangeException if the user must change their password first.
    """
    user_id = request.session.get("user_id")
    if not user_id:
        raise RequiresLoginException()
    user = db.query(User).filter(User.id == user_id, User.status == "active").first()
    if not user:
        raise RequiresLoginException()

    if user.must_change_password:
        raise RequiresPasswordChangeException()

    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Requires current effective user to be admin or superadmin."""
    if user.role not in ("admin", "superadmin"):
        raise HTTPException(status_code=403, detail="Acesso não autorizado")
    return user


def require_superadmin(user: User = Depends(get_actual_user)) -> User:
    """Requires the *actual* logged-in user to be superadmin (not fooled by impersonation)."""
    if user.role != "superadmin":
        raise HTTPException(status_code=403, detail="Área exclusiva do Super Administrador")
    return user
