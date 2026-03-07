"""
AI client for HiringEval — Anthropic Claude integration.

Error hierarchy:
  AIError (base)
    AIConfigError   — API key missing or empty in .env
    AIAuthError     — SDK rejected the key (wrong/revoked)
    AITimeoutError  — API took too long
    AIParsingError  — Model returned non-JSON or unexpected shape
"""

import json
import os
import re
import subprocess
import tempfile

from anthropic import Anthropic, APIConnectionError, APIStatusError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

MODELO = "claude-sonnet-4-6"

# Palavras que indicam critério técnico/hard skill — proibidas no scorecard comportamental
_PALAVRAS_TECNICAS = frozenset({
    # Linguagens de programação
    "python", "java", "javascript", "typescript", "ruby", "golang", "rust",
    "php", "swift", "kotlin", "scala", "perl", "c++", "c#",
    # Bancos de dados
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
    # BI / dados
    "excel", "powerbi", "tableau", "looker", "databricks", "spark",
    # CRM / ERP / plataformas
    "salesforce", "hubspot", "pipedrive", "zendesk", "sap", "erp", "crm",
    # Cloud e infra
    "aws", "azure", "gcp", "cloud", "kubernetes", "docker", "terraform",
    "linux", "unix",
    # Frontend / backend / APIs
    "react", "angular", "vue", "node", "django", "flask", "fastapi",
    "api", "rest", "graphql", "microservices",
    # Design / ferramentas
    "figma", "photoshop", "illustrator", "autocad", "sketch",
    "git", "jira", "confluence",
    # Office técnico
    "office", "powerpoint",
    # Formação e certificações
    "certificação", "certificado", "diploma", "graduação", "formação", "mba",
    # Idiomas
    "inglês", "espanhol", "francês", "alemão", "mandarim", "idioma", "bilíngue",
    # Genéricas técnicas
    "programação", "código", "software", "hardware", "técnico", "técnica",
    "tecnologia", "ferramenta", "linguagem", "plataforma", "infra",
    # ML / IA
    "tensorflow", "pytorch", "nlp", "machine",
    # Metodologias de engenharia
    "devops", "scrum", "kanban",
})


# ── Custom exceptions ──────────────────────────────────────────────────────────

class AIError(Exception):
    """Base class for all AI-related errors."""


class AIConfigError(AIError):
    """ANTHROPIC_API_KEY ausente ou vazia no .env"""


class AIAuthError(AIError):
    """Chave rejeitada pela API (inválida ou revogada)."""


class AITimeoutError(AIError):
    """A requisição demorou mais que o esperado."""


class AIParsingError(AIError):
    """A API retornou conteúdo inesperado (não-JSON ou shape errado)."""


class AITranscriptionError(AIError):
    """Erro durante extração de áudio (ffmpeg) ou transcrição (Whisper/OpenAI)."""


# ── Lazy client ────────────────────────────────────────────────────────────────

_client: "Anthropic | None" = None


def _get_client() -> Anthropic:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or not api_key.strip():
        raise AIConfigError(
            "ANTHROPIC_API_KEY não está configurada. "
            "Copie .env.example para .env e adicione sua chave de API Anthropic."
        )

    _client = Anthropic(api_key=api_key.strip())
    return _client


# ── Lazy OpenAI client (Whisper) ───────────────────────────────────────────────

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    import openai as _openai_sdk

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise AIConfigError(
            "OPENAI_API_KEY não está configurada. "
            "Adicione sua chave da OpenAI ao arquivo .env."
        )

    _openai_client = _openai_sdk.OpenAI(api_key=api_key.strip())
    return _openai_client


def transcrever_audio(video_path: str) -> str:
    """Extrai áudio via ffmpeg e transcreve com OpenAI Whisper.

    Cria um mp3 temporário (32kbps mono) para manter o arquivo abaixo do
    limite de 25 MB do Whisper mesmo em entrevistas de 2+ horas.
    O mp3 é deletado no bloco finally; o vídeo é deletado pelo chamador.
    """
    mp3_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3_path = tmp.name

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "libmp3lame",
                "-ac", "1",
                "-ab", "32k",
                mp3_path,
            ],
            capture_output=True,
            timeout=300,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise AITranscriptionError(
                f"ffmpeg falhou ao extrair áudio (código {result.returncode}): {stderr[:400]}"
            )

        mp3_size_mb = os.path.getsize(mp3_path) / (1024 * 1024)
        if mp3_size_mb > 24.5:
            raise AITranscriptionError(
                f"Áudio comprimido tem {mp3_size_mb:.1f} MB e excede o limite de 25 MB do Whisper. "
                "Tente um arquivo mais curto."
            )

        client = _get_openai_client()
        with open(mp3_path, "rb") as audio_file:
            resposta = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="pt",
                response_format="text",
            )

        transcricao = resposta if isinstance(resposta, str) else resposta.text
        if not transcricao or not transcricao.strip():
            raise AITranscriptionError(
                "Whisper retornou transcrição vazia. Verifique se o arquivo contém áudio audível."
            )

        return transcricao.strip()

    except AIError:
        raise

    except subprocess.TimeoutExpired:
        raise AITranscriptionError(
            "Extração de áudio excedeu o tempo limite (5 min). Tente um arquivo menor."
        )

    except FileNotFoundError:
        raise AITranscriptionError(
            "ffmpeg não encontrado no servidor. Instale o pacote ffmpeg no ambiente."
        )

    except Exception as e:
        raise AITranscriptionError(
            f"Erro inesperado durante a transcrição: {type(e).__name__}: {e}"
        ) from e

    finally:
        if mp3_path and os.path.exists(mp3_path):
            try:
                os.unlink(mp3_path)
            except OSError:
                pass


# ── Utilities ──────────────────────────────────────────────────────────────────

def _extrair_json(texto: str) -> str:
    """Remove markdown code fences and return the raw JSON string."""
    texto = texto.strip()
    if texto.startswith("```"):
        linhas = texto.split("\n")
        fim = next(
            (i for i in range(len(linhas) - 1, 0, -1) if linhas[i].strip() == "```"),
            len(linhas),
        )
        conteudo = linhas[1:fim]
        if conteudo and conteudo[0].strip().lower() in ("json", ""):
            conteudo = conteudo[1:]
        return "\n".join(conteudo).strip()
    return texto


def _numerar_linhas(transcricao: str) -> str:
    """Add line numbers to each line of a transcript."""
    linhas = transcricao.strip().split("\n")
    return "\n".join(f"[{i + 1}] {linha}" for i, linha in enumerate(linhas))


def _ordenar_evidencias(evidencias: list[str]) -> list[str]:
    """Sort evidence citations by their transcript line number ([N] 'quote')."""
    def _linha(ev: str) -> int:
        m = re.match(r"\[(\d+)\]", ev.strip())
        return int(m.group(1)) if m else 999999
    return sorted(evidencias, key=_linha)


def _normalizar_pesos_inplace(criterios: list) -> None:
    """Normalize criterion weights so they sum to exactly 100 (in-place)."""
    total = sum(c["peso"] for c in criterios)
    if total != 100:
        for c in criterios:
            c["peso"] = round(c["peso"] * 100 / total)
        diferenca = 100 - sum(c["peso"] for c in criterios)
        criterios[0]["peso"] += diferenca


def _max_palavras_rubrica(scorecard: dict) -> int:
    """Return the maximum word count across all rubric descriptions in the scorecard.

    Used as the word-count ceiling when regenerating or editing a criterion,
    so new descriptions never exceed the length of existing ones.
    Clamped between 30 and 80 words.
    """
    max_p = 0
    for c in scorecard.get("criterios", []):
        for nivel in c.get("rubrica", {}).values():
            count = len(nivel.split())
            if count > max_p:
                max_p = count
    return max(30, min(max_p, 80))


def _criterios_tecnicos(criterios: list) -> list[str]:
    """
    Returns the names of criteria whose name contains a technical/hard-skill word.
    Used to validate that the scorecard is purely behavioral.
    """
    flagged = []
    for c in criterios:
        nome = c.get("nome", "").lower().strip()
        for token in nome.split():
            if token in _PALAVRAS_TECNICAS:
                flagged.append(c["nome"])
                break
    return flagged


def _chamar_api(prompt: str, max_tokens: int) -> str:
    """
    Call the Anthropic API and return the text of the first content block.
    Translates SDK exceptions to our AIError hierarchy.
    """
    try:
        client = _get_client()
        resposta = client.messages.create(
            model=MODELO,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resposta.content[0].text
    except AIConfigError:
        raise
    except (TypeError, APIStatusError) as e:
        msg = str(e)
        if "authentication" in msg.lower() or "api_key" in msg.lower() or "401" in msg:
            raise AIAuthError(f"Chave de API inválida ou revogada: {msg}") from e
        raise AIError(f"Erro inesperado na API Anthropic: {msg}") from e
    except APITimeoutError as e:
        raise AITimeoutError("A requisição à API expirou. Aguarde alguns segundos e tente novamente.") from e
    except APIConnectionError as e:
        raise AIError(f"Erro de conexão com a API Anthropic: {e}") from e


# ── Public functions ───────────────────────────────────────────────────────────

def gerar_scorecard(nome_cargo: str, descricao_vaga: str) -> dict:
    prompt = f"""Você é um especialista em People & Culture com foco em avaliação comportamental estruturada.
Sua tarefa é criar um scorecard de entrevista baseado EXCLUSIVAMENTE em competências comportamentais, soft skills e fit cultural para o cargo abaixo.

Cargo: {nome_cargo}
Descrição da Vaga:
{descricao_vaga}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGRAS ABSOLUTAS — SIGA SEM EXCEÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROIBIDO incluir qualquer critério que envolva:
✗ Hard skills técnicas (programação, linguagens de programação, ferramentas, plataformas)
✗ Conhecimento de softwares ou sistemas (Excel, Python, SQL, Salesforce, SAP, Power BI, etc.)
✗ Certificações, diplomas ou requisitos de formação acadêmica
✗ Idiomas ou fluência linguística
✗ Qualquer habilidade que se aprende em curso técnico, e não em interações humanas

OBRIGATÓRIO incluir apenas critérios que avaliem:
✓ Comportamentos observáveis em situações reais de trabalho
✓ Soft skills: liderança, comunicação, resolução de conflitos, adaptabilidade, colaboração, etc.
✓ Fit cultural e de valores com a empresa e o time
✓ Mentalidade e forma de pensar (ex: orientação a resultados, pensamento estratégico, dono do negócio)
✓ Dinâmicas interpessoais e de equipe

SOBRE REQUISITOS TÉCNICOS NA DESCRIÇÃO DA VAGA:
Se a vaga mencionar hard skills técnicas (ex: "deve conhecer Python", "experiência com Salesforce", "fluência em inglês"),
IGNORE-OS completamente — esses são pré-requisitos de triagem e NÃO fazem parte da avaliação comportamental.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGRAS DE FORMATO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Retorne EXATAMENTE 5 critérios — nem mais, nem menos.
2. O nome de cada critério deve ser EXATAMENTE UMA PALAVRA em português (substantivo, inicial maiúscula — ex: "Liderança", "Execução", "Comunicação").
3. Os pesos devem ser números inteiros que somem EXATAMENTE 100.
4. As descrições de rubrica devem ser ESPECÍFICAS e COMPORTAMENTAIS para ESTE cargo. Sem frases genéricas.
5. Cada nível deve descrever um comportamento observável E justificar por que corresponde àquela nota.

Escala da rubrica:
1 = Ausência clara do comportamento exigido
2 = Demonstração fraca ou inconsistente
3 = Competência adequada / nível base
4 = Demonstração forte e consistente
5 = Demonstração excepcional, escalável e estratégica

EXEMPLO RUIM (PROIBIDO): "Domina Python e SQL." / "Possui certificação AWS." / "Fluente em inglês."
EXEMPLO BOM para Liderança/5: "Estruturou e escalou equipe com múltiplos níveis de reporte; implementou sistema de metas (OKRs ou equivalente); tomou decisões de contratação e desligamento com base em critérios objetivos; há evidências de outros gestores replicando seu modelo de gestão."

Retorne APENAS JSON válido (sem markdown, sem explicação):
{{
  "criterios": [
    {{
      "nome": "UmaPalavra",
      "peso": 20,
      "rubrica": {{
        "1": "Descrição observável específica para este cargo — nível 1",
        "2": "Descrição observável específica para este cargo — nível 2",
        "3": "Descrição observável específica para este cargo — nível 3",
        "4": "Descrição observável específica para este cargo — nível 4",
        "5": "Descrição observável específica para este cargo — nível 5"
      }}
    }}
  ]
}}"""

    _MAX_TENTATIVAS = 2
    ultimo_erro: Exception | None = None
    dados: dict = {}

    for tentativa in range(1, _MAX_TENTATIVAS + 1):
        texto = _chamar_api(prompt, max_tokens=4096)

        try:
            bruto = _extrair_json(texto)
            dados = json.loads(bruto)
        except json.JSONDecodeError as e:
            ultimo_erro = AIParsingError(f"A IA retornou JSON inválido no scorecard: {e}")
            print(f"[AI PARSING ERROR] tentativa={tentativa} Resposta bruta:\n{texto}")
            continue

        criterios = dados.get("criterios", [])
        if len(criterios) != 5:
            ultimo_erro = AIParsingError(
                f"Esperado 5 critérios, recebido {len(criterios)}."
            )
            continue

        tecnicos = _criterios_tecnicos(criterios)
        if tecnicos:
            ultimo_erro = AIParsingError(
                f"Scorecard contém critérios técnicos/hard-skill: {tecnicos}. "
                "O scorecard deve avaliar somente competências comportamentais."
            )
            print(f"[SCORECARD VALIDAÇÃO] tentativa={tentativa} critérios técnicos detectados: {tecnicos}")
            continue

        # Passou em todas as validações
        ultimo_erro = None
        break

    if ultimo_erro is not None:
        raise ultimo_erro

    criterios = dados["criterios"]
    _normalizar_pesos_inplace(criterios)
    return dados


def regenerar_criterio_ia(
    scorecard: dict,
    indice: int,
    nome_cargo: str,
    descricao_vaga: str,
    criterio_anterior: str,
    max_palavras: int,
) -> dict:
    """Regenerate a single criterion (by index) with AI, keeping the others intact.

    criterio_anterior: name of the criterion being replaced — the new one must differ.
    max_palavras: maximum word count allowed per rubric description level.
    Returns the new criterion dict with the same peso as the original.
    Raises AIParsingError if validation fails after 2 attempts.
    """
    criterios_atuais = scorecard["criterios"]
    outros_nomes = [c["nome"] for i, c in enumerate(criterios_atuais) if i != indice]
    peso_atual = criterios_atuais[indice]["peso"]

    prompt = f"""Você é um especialista em People & Culture com foco em avaliação comportamental estruturada.
Você está atualizando um scorecard de entrevista. Precisa gerar UM NOVO critério comportamental para substituir um existente.

Cargo: {nome_cargo}
Descrição da Vaga:
{descricao_vaga}

CRITÉRIO ANTERIOR A SER SUBSTITUÍDO: "{criterio_anterior}"
IMPORTANTE: O novo critério deve ser DIFERENTE e NÃO PARECIDO com "{criterio_anterior}".
Deve representar um conceito comportamental distinto — não o mesmo tema com outras palavras.

Os outros 4 critérios já definidos no scorecard (NÃO repita nenhum destes nem o critério anterior):
{', '.join(outros_nomes)}

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

    ultimo_erro: Exception | None = None

    for tentativa in range(1, 3):
        texto = _chamar_api(prompt, max_tokens=2048)

        try:
            bruto = _extrair_json(texto)
            novo = json.loads(bruto)
        except json.JSONDecodeError as e:
            ultimo_erro = AIParsingError(f"JSON inválido ao regenerar critério: {e}")
            continue

        nome = novo.get("nome", "")
        if _criterios_tecnicos([{"nome": nome}]):
            ultimo_erro = AIParsingError(f"Critério gerado é técnico/hard-skill: {nome}")
            continue

        rubrica = novo.get("rubrica", {})
        if not all(str(k) in rubrica for k in range(1, 6)):
            ultimo_erro = AIParsingError("Rubrica incompleta ao regenerar critério")
            continue

        ultimo_erro = None
        break

    if ultimo_erro is not None:
        raise ultimo_erro

    novo["peso"] = peso_atual
    return novo


def gerar_rubrica_criterio(
    nome_criterio: str,
    nome_cargo: str,
    descricao_vaga: str,
    outros_criterios: list,
    max_palavras: int,
) -> dict:
    """Generate rubric levels 1-5 for a manually-named criterion.

    max_palavras: maximum word count allowed per description level.
    Returns a rubric dict with keys "1" through "5".
    Raises AIParsingError if the response is invalid.
    """
    outros_str = ", ".join(outros_criterios) if outros_criterios else "N/A"

    prompt = f"""Você é um especialista em People & Culture com foco em avaliação comportamental estruturada.
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

    texto = _chamar_api(prompt, max_tokens=2048)

    try:
        bruto = _extrair_json(texto)
        dados = json.loads(bruto)
    except json.JSONDecodeError as e:
        raise AIParsingError(f"JSON inválido ao gerar rubrica do critério: {e}") from e

    rubrica = dados.get("rubrica", {})
    if not all(str(k) in rubrica for k in range(1, 6)):
        raise AIParsingError("Rubrica incompleta retornada pela IA")

    return rubrica


def avaliar_criterio_unico(criterio: dict, nome_candidato: str, transcricao: str) -> dict:
    """Evaluate a single criterion for a candidate given their transcript.

    Returns one avaliacao dict. Two possible shapes:
    - Scored:   {"criterio", "nota", "peso", "contribuicao", "evidencias", "lacunas"}
    - Unscored: {"criterio", "peso", "sem_evidencia": True, "motivo", "evidencia_esperada"}
    Raises AIParsingError if the response is invalid.
    """
    numerada = _numerar_linhas(transcricao)
    rubrica_txt = "\n".join(f"  {k}: {v}" for k, v in sorted(criterio["rubrica"].items()))
    nome_c = criterio["nome"]
    peso_c = criterio["peso"]

    prompt = f"""Você é um entrevistador especialista em avaliação estruturada de candidatos.
Avalie o candidato abaixo para UM ÚNICO critério comportamental.

Candidato: {nome_candidato}

CRITÉRIO: {nome_c} (Peso: {peso_c}%)
Rubrica:
{rubrica_txt}

=== TRANSCRIÇÃO (com número de linhas) ===
{numerada}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PASSO 1 — Verifique se há evidência real na transcrição para este critério.
Procure falas do candidato que demonstrem diretamente o comportamento descrito na rubrica.

SE houver evidência suficiente → avalie com nota 1–5 e retorne o formato A.
SE NÃO houver evidência suficiente → NÃO atribua nota e retorne o formato B.

REGRA CRÍTICA: Ausência de evidência ≠ desempenho ruim.
Não invente contexto. Não assuma coisas não ditas na transcrição.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO A — com evidência (use quando há base na transcrição para avaliar):
{{
  "criterio": "{nome_c}",
  "nota": 3,
  "peso": {peso_c},
  "contribuicao": {round(3 * peso_c / 5, 1)},
  "evidencias": ["[12] 'citação exata da linha 12'"],
  "lacunas": "O que não foi demonstrado ou estava ausente"
}}

FORMATO B — sem evidência (use quando a transcrição não permite avaliar este critério):
{{
  "criterio": "{nome_c}",
  "peso": {peso_c},
  "sem_evidencia": true,
  "motivo": "Explicação breve de por que a transcrição não permite avaliar este critério.",
  "evidencia_esperada": "Que tipo de falas ou situações seriam necessárias para avaliar este critério."
}}

Retorne APENAS JSON válido (sem markdown, sem explicação), usando exatamente um dos dois formatos acima."""

    texto = _chamar_api(prompt, max_tokens=1024)

    try:
        bruto = _extrair_json(texto)
        dados = json.loads(bruto)
    except json.JSONDecodeError as e:
        raise AIParsingError(f"JSON inválido ao avaliar critério único: {e}") from e

    dados["criterio"] = nome_c
    dados["peso"] = peso_c

    if dados.get("sem_evidencia"):
        dados.pop("nota", None)
        dados.pop("contribuicao", None)
        dados.pop("evidencias", None)
        dados.pop("lacunas", None)
    else:
        dados["contribuicao"] = round((dados["nota"] * peso_c) / 5, 1)
        dados["evidencias"] = _ordenar_evidencias(dados.get("evidencias") or [])

    return dados


def avaliar_candidato(scorecard: dict, nome_candidato: str, transcricao: str) -> dict:
    numerada = _numerar_linhas(transcricao)

    blocos_criterio = []
    for c in scorecard["criterios"]:
        rubrica = "\n".join(f"  {k}: {v}" for k, v in sorted(c["rubrica"].items()))
        blocos_criterio.append(
            f"CRITÉRIO: {c['nome']} (Peso: {c['peso']}%)\nRubrica:\n{rubrica}"
        )
    texto_criterios = "\n\n".join(blocos_criterio)

    prompt = f"""Você é um entrevistador especialista em avaliação estruturada de candidatos. Avalie o candidato abaixo com base no scorecard fornecido.

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

    texto = _chamar_api(prompt, max_tokens=6144)

    try:
        bruto = _extrair_json(texto)
        dados = json.loads(bruto)
    except json.JSONDecodeError as e:
        print(f"[AI PARSING ERROR] Resposta bruta da avaliação:\n{texto}")
        raise AIParsingError(f"A IA retornou JSON inválido na avaliação: {e}") from e

    # Server-side: recalculate contributions and sort evidence; skip sem_evidencia entries
    for av in dados["avaliacoes"]:
        if av.get("sem_evidencia"):
            av.pop("nota", None)
            av.pop("contribuicao", None)
            av.pop("evidencias", None)
            av.pop("lacunas", None)
        else:
            av["contribuicao"] = round((av["nota"] * av["peso"]) / 5, 1)
            av["evidencias"] = _ordenar_evidencias(av.get("evidencias") or [])

    # Normalize nota_final to 0–100 using only scored criteria
    scored = [av for av in dados["avaliacoes"] if not av.get("sem_evidencia")]
    total_peso_scored = sum(av["peso"] for av in scored)
    if total_peso_scored > 0:
        nota_final = round(sum(av["contribuicao"] for av in scored) / total_peso_scored * 100, 1)
    else:
        nota_final = 0.0
    dados["nota_final"] = nota_final
    dados["criterios_avaliados"] = len(scored)
    dados["criterios_total"] = len(dados["avaliacoes"])

    return dados
