import copy
import json
import os
import secrets
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
    RequiresPasswordChangeException,
    authenticate_user,
    get_actual_user,
    get_current_user,
    get_session_user,
    hash_password,
    require_admin,
    require_superadmin,
    verify_password,
)
from models import Candidate, Company, Job, Prompt, SessionLocal, User, create_tables, get_db

SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "hiring-eval-secret-key-change-in-production")


# ── Startup ────────────────────────────────────────────────────────────────────

def _load_prompt(db: Session, name: str) -> str | None:
    """Return the prompt_text for the given name, or None if not found."""
    p = db.query(Prompt).filter(Prompt.name == name).first()
    return p.prompt_text if p else None


# ── Prompt templates (stored in DB; variables use {name} placeholders) ─────────

_PROMPT_GERAR_SCORECARD = """\
Você é um especialista em People & Culture com foco em avaliação comportamental estruturada.
Sua tarefa é criar um scorecard de entrevista para o cargo abaixo.

Cargo: {nome_cargo}
Descrição da Vaga:
{descricao_vaga}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASSO 1 — ANALISE A VAGA ANTES DE GERAR OS CRITÉRIOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Antes de definir os critérios, reflita internamente:
- Quais comportamentos são mais críticos para o sucesso neste cargo específico?
- Quais dinâmicas interpessoais e de tomada de decisão mais diferenciam um profissional mediano de um excelente aqui?
Use essa análise para escolher 5 critérios representativos deste cargo — não critérios genéricos que servem para qualquer vaga.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
O QUE É PROIBIDO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ Hard skills técnicas (programação, linguagens, ferramentas, plataformas)
✗ Conhecimento de softwares ou sistemas (Excel, Python, SQL, Salesforce, SAP, Power BI, etc.)
✗ Certificações, diplomas ou requisitos de formação acadêmica
✗ Idiomas ou fluência linguística
✗ Qualquer habilidade adquirida em curso técnico, e não em interações humanas
✗ Critérios redundantes ou parecidos entre si

Se a vaga mencionar requisitos técnicos (ex: "deve conhecer Python", "fluência em inglês"),
IGNORE-OS — são pré-requisitos de triagem e não fazem parte da avaliação comportamental.

O QUE É OBRIGATÓRIO
✓ Comportamentos observáveis em situações de trabalho
✓ Soft skills: liderança, comunicação, resolução de conflitos, adaptabilidade, colaboração, etc.
✓ Fit cultural, de valores e de mentalidade com o cargo
✓ Dinâmicas interpessoais e forma de tomar decisões

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILOSOFIA DAS RUBRICAS — LEIA COM ATENÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
As rubricas devem avaliar COMO o candidato pensa e responde — não se ele viveu exatamente uma situação específica.
O candidato pode demonstrar a mesma competência com exemplos de contextos completamente diferentes.

❌ ERRADO — exige experiência específica:
"Liderou uma reorganização estratégica do time com múltiplos níveis de reporte."

✅ CERTO — avalia qualidade da resposta:
"Explica decisões considerando impactos de longo prazo e possíveis consequências não óbvias."

Cada nível da rubrica deve descrever SINAIS OBSERVÁVEIS NA RESPOSTA DO CANDIDATO, como:
- Clareza e lógica do raciocínio apresentado
- Capacidade de estruturar o problema antes de explicar a solução
- Qualidade e pertinência dos exemplos usados para sustentar o argumento
- Profundidade da reflexão e consciência das implicações e trade-offs

As descrições devem funcionar independentemente do setor, tamanho de empresa ou cargo anterior do candidato.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGRAS DE FORMATO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Retorne EXATAMENTE 5 critérios — nem mais, nem menos.
2. Cada critério deve medir uma DIMENSÃO COMPORTAMENTAL DIFERENTE (sem critérios parecidos ou redundantes).
3. O nome de cada critério deve ser EXATAMENTE UMA PALAVRA em português (substantivo, inicial maiúscula — ex: "Liderança", "Execução").
4. Os pesos devem ser números inteiros que somem EXATAMENTE 100.
5. Cada descrição de nível deve ter NO MÁXIMO 200 caracteres.
6. Os níveis devem mostrar PROGRESSÃO REAL de comportamento — não apenas variações de intensidade como "fraco / médio / forte".

Escala da rubrica (cada nível descreve um sinal observável na resposta):
1 = Resposta vaga, sem estrutura ou sem exemplos; não demonstra o comportamento
2 = Demonstração superficial ou inconsistente; exemplos genéricos ou sem profundidade
3 = Demonstra o comportamento com clareza em pelo menos um exemplo concreto e bem explicado
4 = Demonstra com consistência, estrutura clara e reflexão sobre impactos, alternativas ou aprendizados
5 = Raciocínio estruturado, exemplos pertinentes e consciência de implicações complexas ou de longo prazo

EXEMPLO RUIM (PROIBIDO):
Liderança/5: "Liderou equipe com múltiplos níveis de reporte e implementou OKRs."
→ Exige experiência específica; candidato de empresa pequena será penalizado injustamente.

EXEMPLO BOM:
Liderança/5: "Explica como influenciou o grupo sem autoridade formal, descreve o raciocínio por trás das decisões e os impactos percebidos."
→ Avalia o pensamento, não o histórico.

Retorne APENAS JSON válido (sem markdown, sem explicação):
{{
  "criterios": [
    {{
      "nome": "UmaPalavra",
      "peso": 20,
      "rubrica": {{
        "1": "Sinal observável na resposta — nível 1 (máx 200 caracteres)",
        "2": "Sinal observável na resposta — nível 2 (máx 200 caracteres)",
        "3": "Sinal observável na resposta — nível 3 (máx 200 caracteres)",
        "4": "Sinal observável na resposta — nível 4 (máx 200 caracteres)",
        "5": "Sinal observável na resposta — nível 5 (máx 200 caracteres)"
      }}
    }}
  ]
}}"""

_PROMPT_REGENERAR_CRITERIO = """\
Você é um especialista em People & Culture com foco em avaliação comportamental estruturada.
Você está atualizando um scorecard de entrevista. Precisa gerar UM NOVO critério comportamental para substituir um existente.

Cargo: {nome_cargo}
Descrição da Vaga:
{descricao_vaga}

CRITÉRIO ANTERIOR A SER SUBSTITUÍDO: "{criterio_anterior}"
IMPORTANTE: O novo critério deve ser DIFERENTE e NÃO PARECIDO com "{criterio_anterior}".
Deve representar um conceito comportamental distinto — não o mesmo tema com outras palavras.

Os outros 4 critérios já definidos no scorecard (NÃO repita nenhum destes nem o critério anterior):
{outros_nomes}

Gere EXATAMENTE 1 critério comportamental novo.

REGRAS ABSOLUTAS:
✗ PROIBIDO: hard skills técnicas, ferramentas, linguagens, certificações, idiomas
✗ PROIBIDO: repetir ou reescrever "{criterio_anterior}" com sinônimos
✓ OBRIGATÓRIO: comportamento observável, soft skill, fit cultural — diferente dos existentes

FORMATO:
- nome: UMA PALAVRA em português (substantivo, inicial maiúscula — ex: "Liderança", "Execução")
- rubrica: 5 níveis descritivos e específicos para este cargo
- Cada descrição de nível deve ter NO MÁXIMO {max_palavras} palavras

Retorne APENAS JSON válido (sem markdown, sem explicação):
{{
  "nome": "UmaPalavra",
  "rubrica": {{
    "1": "Descrição observável específica nível 1",
    "2": "Descrição observável específica nível 2",
    "3": "Descrição observável específica nível 3",
    "4": "Descrição observável específica nível 4",
    "5": "Descrição observável específica nível 5"
  }}
}}"""

_PROMPT_GERAR_RUBRICA = """\
Você é um especialista em People & Culture com foco em avaliação comportamental estruturada.
Gere a rubrica de avaliação para o critério comportamental abaixo, no contexto deste cargo.

Cargo: {nome_cargo}
Descrição da Vaga:
{descricao_vaga}

Critério a descrever: {nome_criterio}
(Outros critérios do scorecard para contexto: {outros_str})

REGRAS:
- Descreva comportamentos observáveis e específicos para este cargo
- Cada nível deve ser distinto e claro
- Escala: 1=ausente, 2=fraco, 3=adequado, 4=forte, 5=excepcional
- Cada descrição de nível deve ter NO MÁXIMO {max_palavras} palavras

Retorne APENAS JSON válido (sem markdown, sem explicação):
{{
  "rubrica": {{
    "1": "Descrição observável nível 1",
    "2": "Descrição observável nível 2",
    "3": "Descrição observável nível 3",
    "4": "Descrição observável nível 4",
    "5": "Descrição observável nível 5"
  }}
}}"""

_PROMPT_AVALIAR_CANDIDATO = """\
Você é um entrevistador especialista em avaliação estruturada de candidatos. Avalie o candidato abaixo com base no scorecard fornecido.

Candidato: {nome_candidato}

=== SCORECARD ===
{texto_criterios}

=== TRANSCRIÇÃO (com número de linhas) ===
{numerada}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES — LEIA COM ATENÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Para CADA critério do scorecard, siga este processo:

PASSO 1 — Verifique se há evidência real na transcrição.
Procure falas do candidato que demonstrem diretamente o comportamento avaliado naquele critério.

SE houver evidência suficiente → avalie com nota 1–5 conforme a rubrica (use o objeto "com nota").
SE NÃO houver evidência suficiente → NÃO atribua nota (use o objeto "sem evidência").

REGRA CRÍTICA: Ausência de evidência NÃO significa desempenho ruim.
Não invente contexto. Não assuma comportamentos não descritos na transcrição.
Só cite linhas que realmente existem na transcrição.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATOS DOS OBJETOS DE AVALIAÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Objeto COM nota (quando há evidência para avaliar):
{{
  "criterio": "NomeDoCriterio",
  "nota": 3,
  "peso": 20,
  "contribuicao": 12.0,
  "evidencias": ["[12] 'citação exata da linha 12'"],
  "lacunas": "O que não foi demonstrado"
}}

Objeto SEM evidência (quando a transcrição não permite avaliar este critério):
{{
  "criterio": "NomeDoCriterio",
  "peso": 20,
  "sem_evidencia": true,
  "motivo": "Explicação breve de por que a transcrição não permite avaliar este critério.",
  "evidencia_esperada": "Que tipo de falas ou situações seriam necessárias para avaliar este critério."
}}

Regras adicionais para objetos COM nota:
- nota: inteiro de 1 a 5, estritamente de acordo com a rubrica
- contribuicao = (nota × peso) / 5
- evidencias: CITAÇÕES DIRETAS com número de linha no formato "[LINHA] 'citação exata'"
- lacunas: o que ficou ausente ou não foi demonstrado

Retorne APENAS JSON válido (sem markdown, sem explicação):
{{
  "avaliacoes": [
    {{ objeto com nota OU sem evidência para cada critério }}
  ]
}}"""


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

    # One-time migration to email-based auth: delete all users and create new superadmin
    gabriel = db.query(User).filter(User.email == "gmakdesiy@gmail.com").first()
    if not gabriel:
        db.query(User).delete()
        db.flush()
        superadmin = User(
            email="gmakdesiy@gmail.com",
            name="Gabriel Makdesi",
            password_hash=hash_password("Cabine2026!"),
            role="superadmin",
            status="active",
            email_verified=True,
            company_id=None,
            is_active=True,
        )
        db.add(superadmin)
        print("=" * 60)
        print("SUPERADMIN CRIADO  →  email: gmakdesiy@gmail.com  |  senha: Cabine2026!")
        print("Altere a senha após o primeiro login!")
        print("=" * 60)

    # Seed AI prompts (only insert if not already present — never overwrite edits)
    _PROMPTS_SEED = [
        ("gerar_scorecard", _PROMPT_GERAR_SCORECARD),
        ("regenerar_criterio_ia", _PROMPT_REGENERAR_CRITERIO),
        ("gerar_rubrica_criterio", _PROMPT_GERAR_RUBRICA),
        ("avaliar_candidato", _PROMPT_AVALIAR_CANDIDATO),
    ]
    inserted = 0
    for name, text in _PROMPTS_SEED:
        if not db.query(Prompt).filter(Prompt.name == name).first():
            db.add(Prompt(name=name, prompt_text=text, version=1))
            inserted += 1
    db.commit()
    print(f"[SEED] Prompts de IA: {inserted} inserido(s), {len(_PROMPTS_SEED) - inserted} já existia(m).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Criando/verificando tabelas...")
    create_tables()
    print("[STARTUP] Tabelas OK. Executando seed...")
    db = SessionLocal()
    try:
        _seed_initial_data(db)
    finally:
        db.close()
    print("[STARTUP] Seed concluído. App pronto.")
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


@app.exception_handler(RequiresPasswordChangeException)
async def needs_password_change_handler(request: Request, exc: RequiresPasswordChangeException):
    return RedirectResponse(url="/trocar-senha-obrigatoria", status_code=303)


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
def pagina_login(request: Request, msg: str = None, db: Session = Depends(get_db)):
    user_id = request.session.get("user_id")
    if user_id:
        user = db.query(User).filter(User.id == user_id, User.status == "active").first()
        if user and not user.must_change_password:
            return RedirectResponse(url="/", status_code=303)
        # Sessão desatualizada (usuário bloqueado/removido) — limpa para evitar loop
        request.session.clear()
    return templates.TemplateResponse("login.html", {"request": request, "msg": msg})


@app.post("/login")
def fazer_login(
    request: Request,
    email: str = Form(...),
    senha: str = Form(...),
    db: Session = Depends(get_db),
):
    user, reason = authenticate_user(db, email, senha)

    if user is None:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Email ou senha incorretos"},
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
                {"request": request, "error": "Empresa bloqueada. Entre em contato com o administrador."},
                status_code=401,
            )

    request.session["user_id"] = user.id
    request.session.pop("impersonating_user_id", None)
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
def fazer_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# ── Forced password change ─────────────────────────────────────────────────────

import string as _string

def _gerar_senha_temporaria() -> str:
    alphabet = _string.ascii_letters + _string.digits
    return "".join(secrets.choice(alphabet) for _ in range(10))


@app.get("/trocar-senha-obrigatoria", response_class=HTMLResponse)
def pagina_trocar_senha_obrigatoria(
    request: Request,
    current_user: User = Depends(get_session_user),
):
    if not current_user.must_change_password:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        "trocar_senha_obrigatoria.html",
        {"request": request, "user": current_user},
    )


@app.post("/trocar-senha-obrigatoria")
def processar_trocar_senha_obrigatoria(
    request: Request,
    nova_senha: str = Form(...),
    confirmar_nova_senha: str = Form(...),
    current_user: User = Depends(get_session_user),
    db: Session = Depends(get_db),
):
    if not current_user.must_change_password:
        return RedirectResponse(url="/", status_code=303)

    errors = []
    if len(nova_senha) < 6:
        errors.append("A senha deve ter pelo menos 6 caracteres.")
    if nova_senha != confirmar_nova_senha:
        errors.append("As senhas não coincidem.")

    if errors:
        return templates.TemplateResponse(
            "trocar_senha_obrigatoria.html",
            {"request": request, "user": current_user, "errors": errors},
            status_code=422,
        )

    user = db.query(User).filter(User.id == current_user.id).first()
    user.password_hash = hash_password(nova_senha)
    user.must_change_password = False
    db.commit()
    return RedirectResponse(url="/", status_code=303)


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
    email: str = Form(...),
    senha: str = Form(...),
    confirmar_senha: str = Form(...),
    db: Session = Depends(get_db),
):
    errors = []
    email_clean = email.strip().lower()

    if senha != confirmar_senha:
        errors.append("As senhas não coincidem.")

    if len(senha) < 6:
        errors.append("A senha deve ter pelo menos 6 caracteres.")

    if db.query(User).filter(User.email == email_clean).first():
        errors.append("Este email já está cadastrado.")

    if errors:
        return templates.TemplateResponse(
            "registro.html",
            {"request": request, "errors": errors,
             "form": {"nome_empresa": nome_empresa, "nome": nome, "email": email}},
            status_code=422,
        )

    # Create new company (pending approval)
    company = db.query(Company).filter(Company.name.ilike(nome_empresa.strip())).first()
    if company is None:
        company = Company(name=nome_empresa.strip(), status="pending")
        db.add(company)
        db.flush()

    new_user = User(
        company_id=company.id,
        email=email_clean,
        name=nome.strip(),
        password_hash=hash_password(senha),
        role="admin",
        status="pending",
        email_verified=True,
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
                if not av.get("sem_evidencia"):
                    av["contribuicao"] = round((av["nota"] * novo_peso) / 5, 1)
        scored = [av for av in nova_eval["avaliacoes"] if not av.get("sem_evidencia")]
        total_peso_scored = sum(av["peso"] for av in scored)
        nova_eval["nota_final"] = round(
            sum(av["contribuicao"] for av in scored) / total_peso_scored * 100, 1
        ) if total_peso_scored > 0 else 0.0
        nova_eval["criterios_avaliados"] = len(scored)
        nova_eval["criterios_total"] = len(nova_eval["avaliacoes"])
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
        scored = [av for av in nova_eval["avaliacoes"] if not av.get("sem_evidencia")]
        total_peso_scored = sum(av["peso"] for av in scored)
        nova_eval["nota_final"] = round(
            sum(av["contribuicao"] for av in scored) / total_peso_scored * 100, 1
        ) if total_peso_scored > 0 else 0.0
        nova_eval["criterios_avaliados"] = len(scored)
        nova_eval["criterios_total"] = len(nova_eval["avaliacoes"])
        candidato.evaluation = nova_eval
        candidato.final_score = nova_eval["nota_final"]
        flag_modified(candidato, "evaluation")


# ── Cargo routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def pagina_inicial(
    request: Request,
    empresa_id: str = None,
    usuario_id: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _empresa_id = int(empresa_id) if empresa_id else None
    _usuario_id = int(usuario_id) if usuario_id else None
    q = db.query(Job).filter(Job.is_deleted == False)  # noqa: E712
    if current_user.role == "superadmin":
        if _empresa_id:
            q = q.filter(Job.company_id == _empresa_id)
    else:
        q = q.filter(Job.company_id == current_user.company_id)
        if _usuario_id:
            q = q.filter(Job.created_by_user_id == _usuario_id)
    cargos = q.order_by(Job.created_at.desc()).all()
    empresas = db.query(Company).order_by(Company.name).all() if current_user.role == "superadmin" else []
    usuarios_empresa = []
    if current_user.role == "admin":
        usuarios_empresa = (
            db.query(User)
            .filter(User.company_id == current_user.company_id, User.status == "active")
            .order_by(User.name)
            .all()
        )
    return templates.TemplateResponse(
        "index.html",
        ctx(
            request, current_user,
            jobs=cargos,
            empresas=empresas,
            empresa_id=_empresa_id,
            usuarios_empresa=usuarios_empresa,
            usuario_id=_usuario_id,
        ),
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
        created_by_user_id=current_user.id,
        responsavel_user_id=current_user.id,
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
    usuarios_empresa = []
    if cargo.company_id:
        usuarios_empresa = (
            db.query(User)
            .filter(User.company_id == cargo.company_id, User.status == "active")
            .order_by(User.name)
            .all()
        )
    return templates.TemplateResponse(
        "job_detail.html",
        ctx(request, current_user, job=cargo, active_candidates=active_candidates, msg=msg, usuarios_empresa=usuarios_empresa),
    )


@app.post("/cargos/{cargo_id}/responsavel")
def atualizar_responsavel(
    cargo_id: int,
    responsavel_user_id: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cargo = _get_job(db, cargo_id, current_user)
    if not cargo:
        return RedirectResponse(url="/", status_code=303)
    novo_id = int(responsavel_user_id)
    # Ensure the chosen user belongs to the same company
    novo_responsavel = db.query(User).filter(User.id == novo_id, User.company_id == cargo.company_id).first()
    if novo_responsavel:
        cargo.responsavel_user_id = novo_id
        db.commit()
    return RedirectResponse(url=f"/cargos/{cargo_id}?msg=responsavel_atualizado", status_code=303)


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
        scorecard = ai_client.gerar_scorecard(
            cargo.name, cargo.description,
            prompt_template=_load_prompt(db, "gerar_scorecard"),
        )
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
            prompt_template=_load_prompt(db, "regenerar_criterio_ia"),
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
            novo_nome.strip(), cargo.name, cargo.description, outros, max_palavras,
            prompt_template=_load_prompt(db, "gerar_rubrica_criterio"),
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
        avaliacao = ai_client.avaliar_candidato(
            cargo.scorecard, candidato.name, transcript,
            prompt_template=_load_prompt(db, "avaliar_candidato"),
        )
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
        avaliacao = ai_client.avaliar_candidato(
            cargo.scorecard, candidato.name, transcript,
            prompt_template=_load_prompt(db, "avaliar_candidato"),
        )
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
    flash_reset = request.session.pop("flash_reset", None)

    return templates.TemplateResponse(
        "admin/dashboard.html",
        ctx(
            request, actual_user,
            empresas=empresas,
            usuarios=usuarios,
            empresa_id=empresa_id,
            status_filter=status_filter,
            selected_company=selected_company,
            flash_reset=flash_reset,
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


@app.post("/admin/usuarios/{user_id}/reset-senha")
def reset_senha_usuario_admin(
    request: Request,
    user_id: int,
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user or user.role == "superadmin":
        return RedirectResponse(url="/admin", status_code=303)
    temp_senha = _gerar_senha_temporaria()
    user.password_hash = hash_password(temp_senha)
    user.must_change_password = True
    db.commit()
    request.session["flash_reset"] = {"nome": user.name, "temp_senha": temp_senha}
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


# ── Admin Prompts routes ────────────────────────────────────────────────────────

@app.get("/admin/prompts", response_class=HTMLResponse)
def listar_prompts(
    request: Request,
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    try:
        prompts = db.query(Prompt).order_by(Prompt.name).all()
    except Exception as e:
        print(f"[PROMPTS] Erro ao listar prompts: {e}")
        prompts = []
    return templates.TemplateResponse(
        "admin/prompts.html",
        ctx(request, actual_user, prompts=prompts),
    )


@app.get("/admin/prompts/{prompt_name}/editar", response_class=HTMLResponse)
def editar_prompt_form(
    request: Request,
    prompt_name: str,
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    try:
        prompt = db.query(Prompt).filter(Prompt.name == prompt_name).first()
    except Exception as e:
        print(f"[PROMPTS] Erro ao buscar prompt '{prompt_name}': {e}")
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' não encontrado")
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' não encontrado")
    flash_saved = request.session.pop("flash_prompt_saved", None)
    return templates.TemplateResponse(
        "admin/prompt_editar.html",
        ctx(request, actual_user, prompt=prompt, flash_saved=flash_saved),
    )


@app.post("/admin/prompts/{prompt_name}/editar")
def salvar_prompt(
    request: Request,
    prompt_name: str,
    prompt_text: str = Form(...),
    actual_user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    try:
        prompt = db.query(Prompt).filter(Prompt.name == prompt_name).first()
    except Exception as e:
        print(f"[PROMPTS] Erro ao buscar prompt '{prompt_name}' para salvar: {e}")
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' não encontrado")
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' não encontrado")
    if prompt_text.strip():
        prompt.prompt_text = prompt_text.strip()
        prompt.version += 1
        prompt.updated_at = datetime.utcnow()
        db.commit()
    request.session["flash_prompt_saved"] = True
    return RedirectResponse(url=f"/admin/prompts/{prompt_name}/editar", status_code=303)


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
    flash_reset = request.session.pop("flash_reset", None)
    return templates.TemplateResponse(
        "empresa/usuarios.html",
        ctx(request, current_user, usuarios=usuarios, flash_reset=flash_reset),
    )


@app.post("/empresa/usuarios")
def criar_usuario_empresa(
    nome: str = Form(...),
    email: str = Form(...),
    senha: str = Form(...),
    perfil: str = Form(...),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if current_user.role == "superadmin":
        return RedirectResponse(url="/admin", status_code=303)

    if perfil not in ("avaliador", "admin"):
        perfil = "avaliador"

    email_clean = email.strip().lower()
    if not db.query(User).filter(User.email == email_clean).first():
        new_user = User(
            company_id=current_user.company_id,
            email=email_clean,
            name=nome.strip(),
            password_hash=hash_password(senha),
            role=perfil,
            status="active",
            email_verified=True,
            is_active=True,
        )
        db.add(new_user)
        db.commit()
    return RedirectResponse(url="/empresa/usuarios", status_code=303)


@app.post("/empresa/usuarios/{user_id}/reset-senha")
def reset_senha_usuario_empresa(
    request: Request,
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
    if not target or target.id == current_user.id:
        return RedirectResponse(url="/empresa/usuarios", status_code=303)
    temp_senha = _gerar_senha_temporaria()
    target.password_hash = hash_password(temp_senha)
    target.must_change_password = True
    db.commit()
    request.session["flash_reset"] = {"nome": target.name, "temp_senha": temp_senha}
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
