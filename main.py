from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

import ai_client
from models import Candidate, Job, create_tables, get_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


app = FastAPI(title="HiringEval", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Cargos ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def pagina_inicial(request: Request, db: Session = Depends(get_db)):
    cargos = db.query(Job).order_by(Job.created_at.desc()).all()
    return templates.TemplateResponse("index.html", {"request": request, "jobs": cargos})


@app.get("/cargos/novo", response_class=HTMLResponse)
def formulario_novo_cargo(request: Request):
    return templates.TemplateResponse("job_new.html", {"request": request})


@app.post("/cargos")
def criar_cargo(
    name: str = Form(...),
    description: str = Form(...),
    db: Session = Depends(get_db),
):
    cargo = Job(name=name.strip(), description=description.strip())
    db.add(cargo)
    db.commit()
    db.refresh(cargo)
    return RedirectResponse(url=f"/cargos/{cargo.id}", status_code=303)


@app.get("/cargos/{cargo_id}", response_class=HTMLResponse)
def detalhe_cargo(
    request: Request,
    cargo_id: int,
    msg: str = None,
    db: Session = Depends(get_db),
):
    cargo = db.query(Job).filter(Job.id == cargo_id).first()
    if not cargo:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        "job_detail.html", {"request": request, "job": cargo, "msg": msg}
    )


@app.post("/cargos/{cargo_id}/gerar-scorecard")
def gerar_scorecard_rota(cargo_id: int, db: Session = Depends(get_db)):
    cargo = db.query(Job).filter(Job.id == cargo_id).first()
    if not cargo:
        return RedirectResponse(url="/", status_code=303)
    try:
        scorecard = ai_client.gerar_scorecard(cargo.name, cargo.description)
        cargo.scorecard = scorecard
        db.commit()
        return RedirectResponse(
            url=f"/cargos/{cargo_id}?msg=scorecard_gerado", status_code=303
        )
    except Exception as e:
        print(f"[erro scorecard] {e}")
        return RedirectResponse(url=f"/cargos/{cargo_id}?msg=erro", status_code=303)


# ── Candidatos ────────────────────────────────────────────────────────────────

@app.get("/cargos/{cargo_id}/candidatos/novo", response_class=HTMLResponse)
def formulario_novo_candidato(
    request: Request, cargo_id: int, db: Session = Depends(get_db)
):
    cargo = db.query(Job).filter(Job.id == cargo_id).first()
    if not cargo or not cargo.scorecard:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)
    return templates.TemplateResponse(
        "candidate_new.html", {"request": request, "job": cargo}
    )


@app.post("/cargos/{cargo_id}/candidatos")
def criar_candidato(
    cargo_id: int,
    name: str = Form(...),
    transcript: str = Form(...),
    db: Session = Depends(get_db),
):
    cargo = db.query(Job).filter(Job.id == cargo_id).first()
    if not cargo or not cargo.scorecard:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)

    candidato = Candidate(
        job_id=cargo_id,
        name=name.strip(),
        transcript=transcript.strip(),
    )
    db.add(candidato)
    db.flush()

    try:
        avaliacao = ai_client.avaliar_candidato(
            cargo.scorecard, candidato.name, transcript
        )
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
        return RedirectResponse(
            url=f"/cargos/{cargo_id}?msg=erro_avaliacao", status_code=303
        )


@app.get("/cargos/{cargo_id}/candidatos/{candidato_id}", response_class=HTMLResponse)
def detalhe_candidato(
    request: Request,
    cargo_id: int,
    candidato_id: int,
    db: Session = Depends(get_db),
):
    candidato = (
        db.query(Candidate)
        .filter(Candidate.id == candidato_id, Candidate.job_id == cargo_id)
        .first()
    )
    if not candidato:
        return RedirectResponse(url=f"/cargos/{cargo_id}", status_code=303)
    return templates.TemplateResponse(
        "candidate_detail.html",
        {"request": request, "candidate": candidato, "job": candidato.job},
    )
