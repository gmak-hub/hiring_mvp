import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

import ai_client
from auth import (
    RequiresLoginException,
    authenticate_user,
    get_actual_user,
    get_current_user,
    hash_password,
    require_admin,
    require_superadmin,
)
from models import Candidate, Company, Job, SessionLocal, User, create_tables, get_db

SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "hiring-eval-secret-key-change-in-production")


# ── Startup ────────────────────────────────────────────────────────────────────

def _seed_initial_data(db: Session) -> None:
    """Create default company, assign orphan jobs, and ensure a superadmin exists."""
    # Create default company if none exist
    if db.query(Company).count() == 0:
        default_company = Company(name="Empresa Padrão")
        db.add(default_company)
        db.flush()
        # Assign any pre-existing jobs (no company) to this company
        db.query(Job).filter(Job.company_id == None).update({"company_id": default_company.id})  # noqa: E711

    # Create superadmin if none exists
    if not db.query(User).filter(User.role == "superadmin").first():
        superadmin = User(
            username="admin",
            name="Super Admin",
            password_hash=hash_password("admin123"),
            role="superadmin",
            company_id=None,
            is_active=True,
        )
        db.add(superadmin)
        print("=" * 60)
        print("SUPERADMIN CRIADO  →  usuário: admin  |  senha: admin123")
        print("Altere a senha após o primeiro login!")
        print("=" * 60)

    db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    db = SessionLocal()
    try:
        _seed_initial_data(db)
    finally:
        db.close()
    yield


# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="HiringEval", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Exception handlers ─────────────────────────────────────────────────────────

@app.exception_handler(RequiresLoginException)
async def needs_login_handler(request: Request, exc: RequiresLoginException):
    return RedirectResponse(url="/login", status_code=303)


@app.exception_handler(403)
async def forbidden_handler(request: Request, exc: HTTPException):
    return RedirectResponse(url="/", status_code=303)


# ── Template context helper ────────────────────────────────────────────────────

def ctx(request: Request, current_user: User, **extra) -> dict:
    """Build the base template context shared by all authenticated pages."""
    return {
        "request": request,
        "user": current_user,
        "is_impersonating": bool(request.session.get("impersonating_user_id")),
        **extra,
    }


# ── Auth routes ────────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
def pagina_login(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
def fazer_login(
    request: Request,
    empresa: str = Form(""),
    usuario: str = Form(...),
    senha: str = Form(...),
    db: Session = Depends(get_db),
):
    user = authenticate_user(db, usuario, senha)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Usuário ou senha incorretos"},
            status_code=401,
        )

    if user.role != "superadmin":
        if not user.company:
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Usuário sem empresa associada. Contate o administrador."},
                status_code=401,
            )
        if empresa.strip().lower() != user.company.name.lower():
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Empresa não corresponde a este usuário"},
                status_code=401,
            )

    request.session["user_id"] = user.id
    request.session.pop("impersonating_user_id", None)
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
def fazer_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# ── Job helpers ────────────────────────────────────────────────────────────────

def _get_job(db: Session, job_id: int, user: User) -> Job | None:
    """Fetch active job visible to this user (enforces company isolation)."""
    q = db.query(Job).filter(Job.id == job_id, Job.is_deleted == False)
    if user.role != "superadmin":
        q = q.filter(Job.company_id == user.company_id)
    return q.first()


# ── Cargo routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def pagina_inicial(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(Job).filter(Job.is_deleted == False)
    if current_user.role != "superadmin":
        q = q.filter(Job.company_id == current_user.company_id)
    cargos = q.order_by(Job.created_at.desc()).all()
    return templates.TemplateResponse("index.html", ctx(request, current_user, jobs=cargos))


@app.get("/cargos/novo", response_class=HTMLResponse)
def formulario_novo_cargo(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    return templates.TemplateResponse("job_new.html", ctx(request, current_user))


@app.post("/cargos")
def criar_cargo(
    request: Request,
    name: str = Form(...),
    description: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = Job(
        name=name.strip(),
        description=description.strip(),
        company_id=current_user.company_id,
    )
    db.add(cargo)
    db.commit()
    db.refresh(cargo)
    return RedirectResponse(url=f"/cargos/{cargo.id}", status_code=303)


@app.get("/cargos/{cargo_id}", response_class=HTMLResponse)
def detalhe_cargo(
    request: Request,
    cargo_id: int,
    msg: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo:
        return RedirectResponse(url="/", status_code=303)
    active_candidates = [c for c in cargo.candidates if not c.is_deleted]
    return templates.TemplateResponse(
        "job_detail.html",
        ctx(request, current_user, job=cargo, active_candidates=active_candidates, msg=msg),
    )


@app.post("/cargos/{cargo_id}/gerar-scorecard")
def gerar_scorecard_rota(
    cargo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo:
        return RedirectResponse(url="/", status_code=303)
    try:
        scorecard = ai_client.gerar_scorecard(cargo.name, cargo.description)
        cargo.scorecard = scorecard
        db.commit()
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg=scorecard_gerado", status_code=303)
    except Exception as e:
        print(f"[erro scorecard] {e}")
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg=erro", status_code=303)


@app.post("/cargos/{cargo_id}/excluir")
def excluir_cargo(
    cargo_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo:
        return RedirectResponse(url="/", status_code=303)
    cargo.is_deleted = True
    cargo.deleted_at = datetime.utcnow()
    db.commit()
    return RedirectResponse(url="/", status_code=303)


# ── Candidate routes ───────────────────────────────────────────────────────────

@app.get("/cargos/{cargo_id}/candidatos/novo", response_class=HTMLResponse)
def formulario_novo_candidato(
    request: Request,
    cargo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo or not cargo.scorecard:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)
    return templates.TemplateResponse("candidate_new.html", ctx(request, current_user, job=cargo))


@app.post("/cargos/{cargo_id}/candidatos")
def criar_candidato(
    cargo_id: int,
    name: str = Form(...),
    transcript: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo or not cargo.scorecard:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)

    candidato = Candidate(job_id=cargo_id, name=name.strip(), transcript=transcript.strip())
    db.add(candidato)
    db.flush()

    try:
        avaliacao = ai_client.avaliar_candidato(cargo.scorecard, candidato.name, transcript)
        candidato.evaluation = avaliacao
        candidato.final_score = avaliacao["nota_final"]
        db.commit()
        db.refresh(candidato)
        return RedirectResponse(
            url=f"/cargos/{cargo_id}/candidatos/{candidato.id}", status_code=303
        )
    except Exception as e:
        print(f"[erro avaliação] {e}")
        db.rollback()
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg=erro_avaliacao", status_code=303)


@app.get("/cargos/{cargo_id}/candidatos/{candidato_id}", response_class=HTMLResponse)
def detalhe_candidato(
    request: Request,
    cargo_id: int,
    candidato_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo:
        return RedirectResponse(url="/", status_code=303)
    candidato = (
        db.query(Candidate)
        .filter(
            Candidate.id == candidato_id,
            Candidate.job_id == cargo_id,
            Candidate.is_deleted == False,
        )
        .first()
    )
    if not candidato:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)
    return templates.TemplateResponse(
        "candidate_detail.html",
        ctx(request, current_user, candidate=candidato, job=cargo),
    )


@app.post("/cargos/{cargo_id}/candidatos/{candidato_id}/excluir")
def excluir_candidato(
    cargo_id: int,
    candidato_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo:
        return RedirectResponse(url="/", status_code=303)
    candidato = (
        db.query(Candidate)
        .filter(Candidate.id == candidato_id, Candidate.job_id == cargo_id, Candidate.is_deleted == False)
        .first()
    )
    if candidato:
        candidato.is_deleted = True
        candidato.deleted_at = datetime.utcnow()
        db.commit()
    return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)


# ── Super Admin routes ─────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
def painel_admin(
    request: Request,
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    empresas = db.query(Company).order_by(Company.created_at.desc()).all()
    usuarios = db.query(User).order_by(User.created_at.desc()).all()
    return templates.TemplateResponse(
        "admin/dashboard.html",
        ctx(request, actual_user, empresas=empresas, usuarios=usuarios),
    )


@app.post("/admin/empresas")
def criar_empresa(
    nome: str = Form(...),
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    if not db.query(Company).filter(Company.name == nome.strip()).first():
        db.add(Company(name=nome.strip()))
        db.commit()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/impersonar/{user_id}")
def impersonar_usuario(
    request: Request,
    user_id: int,
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    target = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not target or target.role == "superadmin":
        return RedirectResponse(url="/admin", status_code=303)
    request.session["impersonating_user_id"] = target.id
    return RedirectResponse(url="/", status_code=303)


@app.post("/admin/parar-impersonar")
def parar_impersonar(request: Request, actual_user: User = Depends(require_superadmin)):
    request.session.pop("impersonating_user_id", None)
    return RedirectResponse(url="/admin", status_code=303)


# ── Company Admin routes ───────────────────────────────────────────────────────

@app.get("/empresa/usuarios", response_class=HTMLResponse)
def pagina_usuarios_empresa(
    request: Request,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if current_user.role == "superadmin":
        return RedirectResponse(url="/admin", status_code=303)
    usuarios = (
        db.query(User)
        .filter(User.company_id == current_user.company_id)
        .order_by(User.created_at.desc())
        .all()
    )
    return templates.TemplateResponse(
        "empresa/usuarios.html",
        ctx(request, current_user, usuarios=usuarios),
    )


@app.post("/empresa/usuarios")
def criar_usuario_empresa(
    nome: str = Form(...),
    usuario: str = Form(...),
    senha: str = Form(...),
    perfil: str = Form(...),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if current_user.role == "superadmin":
        return RedirectResponse(url="/admin", status_code=303)

    # Only allow creating avaliador or admin (no superadmin via company panel)
    if perfil not in ("avaliador", "admin"):
        perfil = "avaliador"

    if not db.query(User).filter(User.username == usuario.strip()).first():
        new_user = User(
            company_id=current_user.company_id,
            username=usuario.strip(),
            name=nome.strip(),
            password_hash=hash_password(senha),
            role=perfil,
            is_active=True,
        )
        db.add(new_user)
        db.commit()
    return RedirectResponse(url="/empresa/usuarios", status_code=303)


@app.post("/empresa/usuarios/{user_id}/desativar")
def desativar_usuario(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if current_user.role == "superadmin":
        return RedirectResponse(url="/admin", status_code=303)
    target = (
        db.query(User)
        .filter(User.id == user_id, User.company_id == current_user.company_id)
        .first()
    )
    if target and target.id != current_user.id:
        target.is_active = False
        db.commit()
    return RedirectResponse(url="/empresa/usuarios", status_code=303)
