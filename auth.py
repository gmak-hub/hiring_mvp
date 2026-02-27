from typing import Optional

from fastapi import Depends, HTTPException, Request
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from models import User, get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.username == username, User.is_active == True).first()
    if not user or not verify_password(password, user.password_hash):
        return None
    return user


# ── Custom exception used to trigger login redirect ───────────────────────────

class RequiresLoginException(Exception):
    pass


# ── FastAPI dependencies ───────────────────────────────────────────────────────

def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """
    Returns the *effective* user for the request.
    If a superadmin is impersonating another user, returns the impersonated user.
    This drives all data-access decisions (company filtering, etc.).
    """
    user_id = request.session.get("user_id")
    if not user_id:
        raise RequiresLoginException()

    impersonating_id = request.session.get("impersonating_user_id")
    if impersonating_id:
        user = db.query(User).filter(User.id == impersonating_id, User.is_active == True).first()
        if not user:
            # Impersonated user no longer valid — clear and fall back to actual user
            request.session.pop("impersonating_user_id", None)
            impersonating_id = None

    if not impersonating_id:
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()

    if not user:
        raise RequiresLoginException()

    return user


def get_actual_user(request: Request, db: Session = Depends(get_db)) -> User:
    """
    Returns the *actual* logged-in user, ignoring any active impersonation.
    Use this for permission checks on admin-only routes.
    """
    user_id = request.session.get("user_id")
    if not user_id:
        raise RequiresLoginException()
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise RequiresLoginException()
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
