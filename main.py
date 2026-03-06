import copy
import json
import os
import tempfile
import traceback
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy.orm.attributes import flag_modified

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

import ai_client
from ai_client import AIAuthError, AIConfigError, AIParsingError, AITimeoutError, AITranscriptionError
from auth import (
    RequiresLoginException,
    authenticate_user,
    get_actual_user,
    get_current_user,
    hash_password,
    require_admin,
    require_superadmin,
    verify_password,
)
from models import Candidate, Company, Job, SessionLocal, User, create_tables, get_db

SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "hiring-eval-secret-key-change-in-production")


# ── Startup ────────────────────────────────────────────────────────────────────

def _seed_initial_data(db: Session) -> None:
    """Ensure Teste company exists, assign orphan jobs to it, and ensure a superadmin exists."""
    # Ensure "Teste" company exists
    teste = db.query(Company).filter(Company.name == "Teste").first()
    if not teste:
        teste = Company(name="Teste", status="active")
        db.add(teste)
        db.flush()

    # Assign any orphan jobs (no company) to "Teste"
    db.query(Job).filter(Job.company_id == None).update(  # noqa: E711
        {"company_id": teste.id}
    )

    if not db.query(User).filter(User.role == "superadmin").first():
        superadmin = User(
            username="admin",
            name="Super Admin",
            password_hash=hash_password("admin123"),
            role="superadmin",
            status="active",
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

app = FastAPI(title="Cabine", lifespan=lifespan)
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


# ── AI error → msg code mapping ───────────────────────────────────────────────

def _ai_msg_code(exc: Exception) -> str:
    """Map an AI exception to a URL msg param."""
    if isinstance(exc, AIConfigError):
        return "api_nao_configurada"
    if isinstance(exc, AIAuthError):
        return "erro_ai_auth"
    if isinstance(exc, AITimeoutError):
        return "erro_ai_timeout"
    if isinstance(exc, AIParsingError):
        return "erro_ai_parsing"
    if isinstance(exc, AITranscriptionError):
        return "erro_transcricao"
    return "erro"


# ── Auth routes ────────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
def pagina_login(request: Request, msg: str = None):
    if request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "msg": msg})


@app.post("/login")
def fazer_login(
    request: Request,
    empresa: str = Form(""),
    usuario: str = Form(...),
    senha: str = Form(...),
    db: Session = Depends(get_db),
):
    user, reason = authenticate_user(db, usuario, senha)

    if user is None:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Usuário ou senha incorretos"},
            status_code=401,
        )

    if reason == "pending":
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Sua conta ainda aguarda aprovação. "
                         "Você será notificado quando o acesso for liberado.",
            },
            status_code=401,
        )

    if reason == "blocked":
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Conta bloqueada. Entre em contato com o administrador."},
            status_code=401,
        )

    # reason == "active" — validate company for non-superadmin
    if user.role != "superadmin":
        if not user.company:
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Usuário sem empresa associada. Contate o administrador."},
                status_code=401,
            )
        if user.company.status != "active":
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Empresa pendente de aprovação ou bloqueada."},
                status_code=401,
            )
        if empresa.strip().lower() != user.company.name.lower():
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Empresa não corresponde a este usuário."},
                status_code=401,
            )

    request.session["user_id"] = user.id
    request.session.pop("impersonating_user_id", None)
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
def fazer_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# ── Public registration ────────────────────────────────────────────────────────

@app.get("/registro", response_class=HTMLResponse)
def pagina_registro(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("registro.html", {"request": request})


@app.post("/registro")
def fazer_registro(
    request: Request,
    nome_empresa: str = Form(...),
    nome: str = Form(...),
    usuario: str = Form(...),
    senha: str = Form(...),
    confirmar_senha: str = Form(...),
    db: Session = Depends(get_db),
):
    errors = []

    if senha != confirmar_senha:
        errors.append("As senhas não coincidem.")

    if len(senha) < 6:
        errors.append("A senha deve ter pelo menos 6 caracteres.")

    if db.query(User).filter(User.username == usuario.strip()).first():
        errors.append(f'O usuário "{usuario.strip()}" já está em uso.')

    if errors:
        return templates.TemplateResponse(
            "registro.html",
            {"request": request, "errors": errors,
             "form": {"nome_empresa": nome_empresa, "nome": nome, "usuario": usuario}},
            status_code=422,
        )

    # Find existing company or create a new pending one
    company = db.query(Company).filter(Company.name.ilike(nome_empresa.strip())).first()
    if company is None:
        company = Company(name=nome_empresa.strip(), status="pending")
        db.add(company)
        db.flush()

    new_user = User(
        company_id=company.id,
        username=usuario.strip(),
        name=nome.strip(),
        password_hash=hash_password(senha),
        role="admin",
        status="pending",
        is_active=False,
    )
    db.add(new_user)
    db.commit()

    return RedirectResponse(url="/login?msg=cadastro_enviado", status_code=303)


# ── Account management ─────────────────────────────────────────────────────────

@app.get("/conta", response_class=HTMLResponse)
def pagina_conta(
    request: Request,
    current_user: User = Depends(get_actual_user),
    msg: str = None,
):
    return templates.TemplateResponse("conta.html", ctx(request, current_user, msg=msg))


@app.post("/conta")
def atualizar_conta(
    request: Request,
    nome: str = Form(...),
    senha_atual: str = Form(""),
    nova_senha: str = Form(""),
    confirmar_nova_senha: str = Form(""),
    current_user: User = Depends(get_actual_user),
    db: Session = Depends(get_db),
):
    errors = []

    # Reload user from DB to make changes
    user = db.query(User).filter(User.id == current_user.id).first()

    # Update name
    if nome.strip():
        user.name = nome.strip()

    # Change password (only if fields are filled)
    if nova_senha or senha_atual:
        if not senha_atual:
            errors.append("Informe a senha atual para trocar a senha.")
        elif not verify_password(senha_atual, user.password_hash):
            errors.append("Senha atual incorreta.")
        elif len(nova_senha) < 6:
            errors.append("A nova senha deve ter pelo menos 6 caracteres.")
        elif nova_senha != confirmar_nova_senha:
            errors.append("As novas senhas não coincidem.")
        else:
            user.password_hash = hash_password(nova_senha)

    if errors:
        db.rollback()
        return templates.TemplateResponse(
            "conta.html",
            ctx(request, current_user, errors=errors),
            status_code=422,
        )

    db.commit()
    return RedirectResponse(url="/conta?msg=salvo", status_code=303)


# ── Job helpers ────────────────────────────────────────────────────────────────

def _get_job(db: Session, job_id: int, user: User) -> "Job | None":
    """Fetch active job visible to this user (enforces company isolation)."""
    q = db.query(Job).filter(Job.id == job_id, Job.is_deleted == False)  # noqa: E712
    if user.role != "superadmin":
        q = q.filter(Job.company_id == user.company_id)
    return q.first()


# ── Scorecard candidate helpers ────────────────────────────────────────────────

def _recalcular_notas_candidatos(cargo: "Job", novo_scorecard: dict) -> None:
    """Recalculate candidate scores mathematically when only weights changed (Case A).

    Keeps existing criterion notes; updates peso, contribuicao and nota_final.
    """
    pesos_por_nome = {c["nome"]: c["peso"] for c in novo_scorecard["criterios"]}
    for candidato in cargo.candidates:
        if candidato.is_deleted or candidato.evaluation is None:
            continue
        nova_eval = copy.deepcopy(candidato.evaluation)
        for av in nova_eval["avaliacoes"]:
            novo_peso = pesos_por_nome.get(av["criterio"])
            if novo_peso is not None:
                av["peso"] = novo_peso
                av["contribuicao"] = round((av["nota"] * novo_peso) / 5, 1)
        nova_eval["nota_final"] = round(
            sum(av["contribuicao"] for av in nova_eval["avaliacoes"]), 1
        )
        candidato.evaluation = nova_eval
        candidato.final_score = nova_eval["nota_final"]
        flag_modified(candidato, "evaluation")


def _reavaliar_criterios_alterados(cargo: "Job", scorecard: dict, pendentes: list) -> None:
    """Re-evaluate only the changed criteria per candidate (Case B partial).

    For each candidate, re-evaluates only the criteria listed in `pendentes` using AI,
    keeping all other criteria scores intact, then recalculates the total score.
    Errors per criterion per candidate are logged but do not abort the loop.
    """
    for candidato in cargo.candidates:
        if candidato.is_deleted or candidato.evaluation is None:
            continue
        nova_eval = copy.deepcopy(candidato.evaluation)
        for entrada in pendentes:
            indice = entrada["indice"]
            nome_anterior = entrada["nome_anterior"]
            novo_criterio = scorecard["criterios"][indice]
            try:
                nova_av = ai_client.avaliar_criterio_unico(
                    novo_criterio, candidato.name, candidato.transcript
                )
                # Replace the evaluation entry that matches the old criterion name
                substituido = False
                for j, av in enumerate(nova_eval["avaliacoes"]):
                    if av["criterio"] == nome_anterior:
                        nova_eval["avaliacoes"][j] = nova_av
                        substituido = True
                        break
                if not substituido and indice < len(nova_eval["avaliacoes"]):
                    nova_eval["avaliacoes"][indice] = nova_av
            except Exception as e:
                print(
                    f"[REAVALIAR PARCIAL] candidato_id={candidato.id} "
                    f"criterio={novo_criterio['nome']}: {type(e).__name__}: {e}"
                )
                print(traceback.format_exc())
        nova_eval["nota_final"] = round(
            sum(av["contribuicao"] for av in nova_eval["avaliacoes"]), 1
        )
        candidato.evaluation = nova_eval
        candidato.final_score = nova_eval["nota_final"]
        flag_modified(candidato, "evaluation")


# ── Cargo routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def pagina_inicial(
    request: Request,
    empresa_id: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _empresa_id = int(empresa_id) if empresa_id else None
    q = db.query(Job).filter(Job.is_deleted == False)  # noqa: E712
    if current_user.role != "superadmin":
        q = q.filter(Job.company_id == current_user.company_id)
    elif _empresa_id:
        q = q.filter(Job.company_id == _empresa_id)
    cargos = q.order_by(Job.created_at.desc()).all()
    empresas = db.query(Company).order_by(Company.name).all() if current_user.role == "superadmin" else []
    return templates.TemplateResponse(
        "index.html",
        ctx(request, current_user, jobs=cargos, empresas=empresas, empresa_id=_empresa_id),
    )


@app.get("/cargos/novo", response_class=HTMLResponse)
def formulario_novo_cargo(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    empresas = db.query(Company).order_by(Company.name).all() if current_user.role == "superadmin" else []
    return templates.TemplateResponse("job_new.html", ctx(request, current_user, empresas=empresas))


@app.post("/cargos")
def criar_cargo(
    name: str = Form(...),
    description: str = Form(...),
    company_id: str = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role == "superadmin" and company_id:
        _company_id = int(company_id)
    else:
        _company_id = current_user.company_id
    cargo = Job(
        name=name.strip(),
        description=description.strip(),
        company_id=_company_id,
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
        code = _ai_msg_code(e)
        print(f"[SCORECARD ERROR] type={type(e).__name__} code={code} cargo_id={cargo_id}")
        print(traceback.format_exc())
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg={code}", status_code=303)


@app.post("/cargos/{cargo_id}/scorecard/regenerar-criterio")
def regenerar_criterio_rota(
    cargo_id: int,
    criterio_idx: int = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo or not cargo.scorecard:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)
    try:
        criterio_anterior = cargo.scorecard["criterios"][criterio_idx]["nome"]
        max_palavras = ai_client._max_palavras_rubrica(cargo.scorecard)
        novo = ai_client.regenerar_criterio_ia(
            cargo.scorecard, criterio_idx, cargo.name, cargo.description,
            criterio_anterior, max_palavras,
        )
        novo_scorecard = copy.deepcopy(cargo.scorecard)
        novo_scorecard["criterios"][criterio_idx] = novo
        ai_client._normalizar_pesos_inplace(novo_scorecard["criterios"])
        # Track pending change — keep nome_anterior from first edit if already listed
        pendentes = novo_scorecard.get("_pendentes", [])
        if not any(e["indice"] == criterio_idx for e in pendentes):
            pendentes.append({"indice": criterio_idx, "nome_anterior": criterio_anterior})
        novo_scorecard["_pendentes"] = pendentes
        cargo.scorecard = novo_scorecard
        flag_modified(cargo, "scorecard")
        db.commit()
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg=criterio_regenerado", status_code=303)
    except Exception as e:
        code = _ai_msg_code(e)
        print(f"[REGENERAR CRITERIO ERROR] type={type(e).__name__} code={code}")
        print(traceback.format_exc())
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg={code}", status_code=303)


@app.post("/cargos/{cargo_id}/scorecard/editar-criterio")
def editar_criterio_rota(
    cargo_id: int,
    criterio_idx: int = Form(...),
    novo_nome: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo or not cargo.scorecard:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)
    try:
        criterios = cargo.scorecard["criterios"]
        nome_anterior = criterios[criterio_idx]["nome"]
        outros = [c["nome"] for i, c in enumerate(criterios) if i != criterio_idx]
        max_palavras = ai_client._max_palavras_rubrica(cargo.scorecard)
        nova_rubrica = ai_client.gerar_rubrica_criterio(
            novo_nome.strip(), cargo.name, cargo.description, outros, max_palavras
        )
        novo_scorecard = copy.deepcopy(cargo.scorecard)
        novo_scorecard["criterios"][criterio_idx]["nome"] = novo_nome.strip()
        novo_scorecard["criterios"][criterio_idx]["rubrica"] = nova_rubrica
        # Track pending change — keep nome_anterior from first edit if already listed
        pendentes = novo_scorecard.get("_pendentes", [])
        if not any(e["indice"] == criterio_idx for e in pendentes):
            pendentes.append({"indice": criterio_idx, "nome_anterior": nome_anterior})
        novo_scorecard["_pendentes"] = pendentes
        cargo.scorecard = novo_scorecard
        flag_modified(cargo, "scorecard")
        db.commit()
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg=criterio_editado", status_code=303)
    except Exception as e:
        code = _ai_msg_code(e)
        print(f"[EDITAR CRITERIO ERROR] type={type(e).__name__} code={code}")
        print(traceback.format_exc())
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg={code}", status_code=303)


@app.post("/cargos/{cargo_id}/scorecard/atualizar-pesos")
def atualizar_pesos_rota(
    cargo_id: int,
    pesos_json: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo or not cargo.scorecard:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)
    try:
        lista_pesos = json.loads(pesos_json)
        n = len(cargo.scorecard["criterios"])
        if len(lista_pesos) != n:
            return RedirectResponse(url=f"/cargos/{cargo_id}?msg=erro", status_code=303)
        novo_scorecard = copy.deepcopy(cargo.scorecard)
        for i, p in enumerate(lista_pesos):
            novo_scorecard["criterios"][i]["peso"] = max(1, int(p))
        ai_client._normalizar_pesos_inplace(novo_scorecard["criterios"])
        novo_scorecard["criterios"].sort(key=lambda c: c["peso"], reverse=True)
        cargo.scorecard = novo_scorecard
        flag_modified(cargo, "scorecard")
        _recalcular_notas_candidatos(cargo, novo_scorecard)
        db.commit()
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg=pesos_atualizados", status_code=303)
    except Exception:
        print(traceback.format_exc())
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg=erro", status_code=303)


@app.post("/cargos/{cargo_id}/scorecard/salvar-alteracoes")
def salvar_alteracoes_rota(
    cargo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo or not cargo.scorecard:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)
    try:
        pendentes = cargo.scorecard.get("_pendentes", [])
        if pendentes:
            _reavaliar_criterios_alterados(cargo, cargo.scorecard, pendentes)
        novo_scorecard = copy.deepcopy(cargo.scorecard)
        novo_scorecard.pop("_pendentes", None)
        cargo.scorecard = novo_scorecard
        flag_modified(cargo, "scorecard")
        db.commit()
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg=alteracoes_salvas", status_code=303)
    except Exception as e:
        code = _ai_msg_code(e)
        print(f"[SALVAR ALTERACOES ERROR] type={type(e).__name__} code={code}")
        print(traceback.format_exc())
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg={code}", status_code=303)


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
    msg: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo or not cargo.scorecard:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)
    return templates.TemplateResponse("candidate_new.html", ctx(request, current_user, job=cargo, msg=msg))


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
        code = _ai_msg_code(e)
        print(f"[AVALIAÇÃO ERROR] type={type(e).__name__} code={code} cargo_id={cargo_id}")
        print(traceback.format_exc())
        db.rollback()
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg={code}", status_code=303)


@app.post("/cargos/{cargo_id}/candidatos/video")
def criar_candidato_video(
    cargo_id: int,
    name: str = Form(...),
    video: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo or not cargo.scorecard:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)

    ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4a"}
    LIMITE_BYTES = 500 * 1024 * 1024  # 500 MB

    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return RedirectResponse(
            url=f"/cargos/{cargo_id}/candidatos/novo?msg=formato_invalido",
            status_code=303,
        )

    video_path: str | None = None
    transcript: str | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            video_path = tmp.name
            tamanho = 0
            while True:
                chunk = video.file.read(256 * 1024)  # 256 KB chunks
                if not chunk:
                    break
                tamanho += len(chunk)
                if tamanho > LIMITE_BYTES:
                    raise AITranscriptionError("O arquivo enviado excede o limite de 500 MB.")
                tmp.write(chunk)

        transcript = ai_client.transcrever_audio(video_path)

    except Exception as e:
        code = _ai_msg_code(e)
        print(f"[VIDEO TRANSCRIÇÃO ERROR] type={type(e).__name__} code={code} cargo_id={cargo_id}")
        print(traceback.format_exc())
        return RedirectResponse(
            url=f"/cargos/{cargo_id}/candidatos/novo?msg={code}", status_code=303
        )
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except OSError:
                pass

    candidato = Candidate(job_id=cargo_id, name=name.strip(), transcript=transcript)
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
        code = _ai_msg_code(e)
        print(f"[VIDEO AVALIAÇÃO ERROR] type={type(e).__name__} code={code} cargo_id={cargo_id}")
        print(traceback.format_exc())
        db.rollback()
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg={code}", status_code=303)


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
            Candidate.is_deleted == False,  # noqa: E712
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
        .filter(
            Candidate.id == candidato_id,
            Candidate.job_id == cargo_id,
            Candidate.is_deleted == False,  # noqa: E712
        )
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
    empresa_id: int = None,
    status_filter: str = "todos",
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    empresas = db.query(Company).order_by(Company.created_at.desc()).all()

    # Build user query with optional filters
    q = db.query(User).order_by(User.created_at.desc())
    if empresa_id:
        q = q.filter(User.company_id == empresa_id)
    if status_filter != "todos":
        q = q.filter(User.status == status_filter)
    usuarios = q.all()

    selected_company = db.query(Company).filter(Company.id == empresa_id).first() if empresa_id else None

    return templates.TemplateResponse(
        "admin/dashboard.html",
        ctx(
            request, actual_user,
            empresas=empresas,
            usuarios=usuarios,
            empresa_id=empresa_id,
            status_filter=status_filter,
            selected_company=selected_company,
        ),
    )


@app.post("/admin/empresas")
def criar_empresa(
    nome: str = Form(...),
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    if not db.query(Company).filter(Company.name == nome.strip()).first():
        db.add(Company(name=nome.strip(), status="active"))
        db.commit()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/empresas/{empresa_id}/status")
def mudar_status_empresa(
    empresa_id: int,
    novo_status: str = Form(...),
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    empresa = db.query(Company).filter(Company.id == empresa_id).first()
    if empresa and novo_status in ("pending", "active", "blocked"):
        empresa.status = novo_status
        # When approving a company, also approve its pending admin user
        if novo_status == "active":
            db.query(User).filter(
                User.company_id == empresa_id,
                User.role == "admin",
                User.status == "pending",
            ).update({"status": "active", "is_active": True})
        db.commit()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/empresas/{empresa_id}/editar")
def editar_empresa(
    empresa_id: int,
    nome: str = Form(...),
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    empresa = db.query(Company).filter(Company.id == empresa_id).first()
    if empresa and nome.strip():
        # Check name uniqueness (excluding self)
        dup = db.query(Company).filter(
            Company.name == nome.strip(), Company.id != empresa_id
        ).first()
        if not dup:
            empresa.name = nome.strip()
            db.commit()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/usuarios/{user_id}/status")
def mudar_status_usuario(
    user_id: int,
    novo_status: str = Form(...),
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if user and user.role != "superadmin" and novo_status in ("pending", "active", "blocked"):
        user.status = novo_status
        user.is_active = novo_status == "active"
        db.commit()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/usuarios/{user_id}/excluir")
def excluir_usuario_permanente(
    user_id: int,
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if user and user.role != "superadmin" and user.status == "blocked":
        db.delete(user)
        db.commit()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/empresas/{empresa_id}/excluir")
def excluir_empresa_permanente(
    empresa_id: int,
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    empresa = db.query(Company).filter(Company.id == empresa_id).first()
    if empresa and empresa.status == "blocked":
        job_ids = [j.id for j in db.query(Job).filter(Job.company_id == empresa_id).all()]
        if job_ids:
            db.query(Candidate).filter(Candidate.job_id.in_(job_ids)).delete(synchronize_session=False)
            db.query(Job).filter(Job.company_id == empresa_id).delete(synchronize_session=False)
        db.query(User).filter(User.company_id == empresa_id).delete(synchronize_session=False)
        db.delete(empresa)
        db.commit()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/impersonar/{user_id}")
def impersonar_usuario(
    request: Request,
    user_id: int,
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    target = db.query(User).filter(User.id == user_id, User.status == "active").first()
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

    if perfil not in ("avaliador", "admin"):
        perfil = "avaliador"

    if not db.query(User).filter(User.username == usuario.strip()).first():
        new_user = User(
            company_id=current_user.company_id,
            username=usuario.strip(),
            name=nome.strip(),
            password_hash=hash_password(senha),
            role=perfil,
            status="active",
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
        target.status = "blocked"
        target.is_active = False
        db.commit()
    return RedirectResponse(url="/empresa/usuarios", status_code=303)
