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


def _ordenar_evidencias(evidencias: list) -> list:
    """Sort evidence items by transcript line number.

    Each item is a dict {"trecho": "[N] '...'", "interpretacao": "..."}.
    """
    def _linha(ev) -> int:
        trecho = ev.get("trecho", "") if isinstance(ev, dict) else str(ev)
        m = re.match(r"\[(\d+)\]", trecho.strip())
        return int(m.group(1)) if m else 999999
    return sorted(evidencias, key=_linha)


def _extrair_momentos(transcricao: str, nome_candidato: str) -> list[dict]:
    """Stage 1 of evaluation: read the full transcript and extract relevant behavioral moments.

    Returns a list of dicts: {"tema": str, "trecho": str, "resumo": str}
    - tema:   behavioral theme (e.g. "tomada de decisão", "gestão de conflitos")
    - trecho: 1-2 sentences from the candidate with transcript line number
    - resumo: one objective sentence about what the moment reveals
    Raises AIParsingError if the response cannot be parsed or is empty.
    """
    numerada = _numerar_linhas(transcricao)

    prompt = f"""Você é um analista de entrevistas comportamentais.
Leia a transcrição completa da entrevista abaixo e extraia os momentos mais relevantes da conversa.

Candidato: {nome_candidato}

=== TRANSCRIÇÃO (com número de linhas) ===
{numerada}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Leia a entrevista inteira como um fluxo contínuo de conversa — não como frases isoladas.
O raciocínio do candidato muitas vezes se constrói ao longo de vários turnos:
pergunta → resposta → follow-up do entrevistador → continuação da resposta.
Trate turnos relacionados como parte do mesmo momento quando fizerem sentido como sequência.

Depois de ler tudo, identifique os momentos mais reveladores — trechos onde o candidato:
- Descreve uma situação, decisão ou ação que tomou
- Revela como pensa, raciocina ou estrutura um problema
- Demonstra uma habilidade interpessoal, de liderança ou colaboração
- Responde a um follow-up de forma que revela mais sobre seu comportamento ou valores

Para cada momento, extraia:
- TEMA: o tema comportamental daquele momento (ex: "resolução de conflitos", "tomada de decisão", "gestão de time", "adaptabilidade", "comunicação sob pressão")
- TRECHO: 1 ou 2 frases do candidato que representem o núcleo daquele momento, com número de linha no formato "[LINHA] '...'"
- RESUMO: uma frase objetiva explicando o que esse momento revela sobre o candidato

REGRAS:
- Capture momentos que representem DIFERENTES aspectos do comportamento do candidato
- Priorize trechos em que o candidato explica raciocínio ou impacto, não apenas lista fatos
- Ignore turnos puramente técnicos (ferramentas, sistemas, código) ou logísticos (datas, cargos, nomes de empresa)
- Cada trecho deve conter no máximo 1 ou 2 frases do candidato

Retorne APENAS JSON válido (sem markdown, sem explicação):
{{
  "momentos": [
    {{
      "tema": "tema comportamental do momento",
      "trecho": "[N] 'citação direta do candidato — 1 ou 2 frases'",
      "resumo": "Uma frase objetiva sobre o que esse momento revela."
    }}
  ]
}}"""

    texto = _chamar_api(prompt, max_tokens=4096)

    try:
        bruto = _extrair_json(texto)
        dados = json.loads(bruto)
    except json.JSONDecodeError as e:
        raise AIParsingError(f"JSON inválido ao extrair momentos da entrevista: {e}") from e

    momentos = dados.get("momentos", [])
    if not momentos:
        raise AIParsingError("Extração de momentos retornou lista vazia.")

    return momentos


def _formatar_momentos(momentos: list[dict]) -> str:
    """Format extracted moments into a readable block for evaluation prompts."""
    linhas = []
    for i, m in enumerate(momentos, 1):
        linhas.append(f"Momento {i} — Tema: {m['tema']}")
        linhas.append(f"  Trecho:  {m['trecho']}")
        linhas.append(f"  Resumo:  {m['resumo']}")
    return "\n".join(linhas)


def _extrair_evidencias_por_criterio(
    criterios: list[dict],
    transcricao: str,
    nome_candidato: str,
) -> dict[str, dict]:
    """Stage 1 of evaluation: extract positive and negative evidence for each criterion.

    Returns a dict keyed by criterion name:
    {
      "NomeCriterio": {
        "positivas": [{"trecho": "[N] '...'", "interpretacao": "..."}, ...],
        "negativas": [{"trecho": "[N] '...'", "interpretacao": "..."}, ...]
      }
    }
    Raises AIParsingError if the response cannot be parsed.
    """
    numerada = _numerar_linhas(transcricao)

    blocos = []
    for c in criterios:
        rubrica_txt = "\n".join(f"    {k}: {v}" for k, v in sorted(c["rubrica"].items()))
        blocos.append(f"  Critério: {c['nome']}\n  Rubrica:\n{rubrica_txt}")
    criterios_txt = "\n\n".join(blocos)

    prompt = f"""Você é um analista de entrevistas comportamentais.
Sua tarefa é identificar evidências na transcrição para cada critério do scorecard.
NÃO dê notas nesta etapa — apenas colete sinais observáveis.

Candidato: {nome_candidato}

=== SCORECARD ===
{criterios_txt}

=== TRANSCRIÇÃO (com número de linhas) ===
{numerada}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Leia a transcrição inteira. Depois, para CADA critério do scorecard:

PASSO 1 — Identifique sinais POSITIVOS
Falas do candidato que demonstram o comportamento descrito na rubrica daquele critério.

PASSO 2 — Identifique sinais NEGATIVOS
Falas que revelam ausência, inconsistência ou fraqueza naquele comportamento.

Para cada evidência (positiva ou negativa):
- trecho: citação direta de 1 ou 2 frases do candidato, com número de linha no formato "[LINHA] '...'"
- interpretacao: uma frase curta explicando o que aquela fala demonstra ou revela

LIMITES:
- Extraia TODOS os sinais que encontrar — não limite a quantidade
- Se não houver, retorne lista vazia para aquela categoria
- Use apenas o que realmente aparece na transcrição — PROIBIDO inventar evidências
- PROIBIDO dar nota ou fazer recomendação nesta etapa

Retorne APENAS JSON válido (sem markdown, sem explicação):
{{
  "evidencias": {{
    "NomeCriterio": {{
      "positivas": [
        {{"trecho": "[N] 'citação direta do candidato'", "interpretacao": "O que essa fala demonstra positivamente."}}
      ],
      "negativas": [
        {{"trecho": "[N] 'citação direta do candidato'", "interpretacao": "O que essa fala revela como ausência ou fraqueza."}}
      ]
    }}
  }}
}}

Use os nomes exatos dos critérios do scorecard como chaves. Inclua todos os critérios, mesmo os que não tiveram evidências (retorne listas vazias)."""

    texto = _chamar_api(prompt, max_tokens=4096)

    try:
        bruto = _extrair_json(texto)
        dados = json.loads(bruto)
    except json.JSONDecodeError as e:
        raise AIParsingError(f"JSON inválido ao extrair evidências por critério: {e}") from e

    return dados.get("evidencias", {})


# Minimum number of valid evidence items required to score a criterion.
# If a scored criterion has fewer than this, it is demoted to sem_evidencia server-side.
_MIN_EVIDENCIAS = 2


def _evidencia_valida(ev) -> bool:
    """Return True if an evidence item contains a real, non-trivial transcript excerpt."""
    if not isinstance(ev, dict):
        return False
    trecho = ev.get("trecho", "").strip()
    return len(trecho) > 10


def _aplicar_minimo_evidencias(av: dict) -> dict:
    """Enforce the minimum evidence rule on a single scored avaliacao.

    If the criterion is already sem_evidencia, it is returned unchanged.
    If it is scored but has fewer than _MIN_EVIDENCIAS valid evidence items,
    it is converted to sem_evidencia with an explanatory motivo.
    """
    if av.get("sem_evidencia"):
        return av

    validas = [ev for ev in (av.get("evidencias") or []) if _evidencia_valida(ev)]
    if len(validas) >= _MIN_EVIDENCIAS:
        return av

    nome_c = av.get("criterio", "este critério")
    return {
        "criterio": av["criterio"],
        "peso": av["peso"],
        "sem_evidencia": True,
        "motivo": (
            f"Apenas {len(validas)} evidência(s) com trecho explícito encontrada(s) "
            f"para {nome_c}. São necessárias no mínimo {_MIN_EVIDENCIAS} para gerar uma nota."
        ),
        "evidencia_esperada": (
            av.get("lacunas")
            or "Mais trechos diretos da entrevista demonstrando este comportamento."
        ),
    }


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

    Uses a two-stage process:
    - Stage 1: extract positive and negative evidence for this criterion from the transcript
    - Stage 2: score the criterion using only the extracted evidence

    Returns one avaliacao dict. Two possible shapes:
    - Scored:   {"criterio", "nota", "peso", "contribuicao", "evidencias", "lacunas"}
    - Unscored: {"criterio", "peso", "sem_evidencia": True, "motivo", "evidencia_esperada"}
    Raises AIParsingError if the response is invalid.
    """
    nome_c = criterio["nome"]
    peso_c = criterio["peso"]
    rubrica_txt = "\n".join(f"  {k}: {v}" for k, v in sorted(criterio["rubrica"].items()))

    # Stage 1 — extract positive and negative evidence for this criterion
    ev_por_criterio = _extrair_evidencias_por_criterio([criterio], transcricao, nome_candidato)
    ev = ev_por_criterio.get(nome_c, {})
    positivas = ev.get("positivas", [])
    negativas = ev.get("negativas", [])

    pos_txt = (
        "\n".join(f"  + {e['trecho']} — {e['interpretacao']}" for e in positivas)
        or "  (nenhuma)"
    )
    neg_txt = (
        "\n".join(f"  - {e['trecho']} — {e['interpretacao']}" for e in negativas)
        or "  (nenhuma)"
    )

    # Stage 2 — score using only the extracted evidence
    prompt = f"""Você é um entrevistador especialista em avaliação estruturada de candidatos.
Avalie o candidato para o critério abaixo usando APENAS as evidências fornecidas.
Não busque novas evidências — use somente o que está listado.

Candidato: {nome_candidato}

CRITÉRIO: {nome_c} (Peso: {peso_c}%)
Rubrica:
{rubrica_txt}

Evidências positivas ({len(positivas)}):
{pos_txt}

Evidências negativas ({len(negativas)}):
{neg_txt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use APENAS as evidências acima. Não busque novas na transcrição.

SE houver evidências suficientes → avalie com nota 1–5 e retorne o FORMATO A.
SE não houver evidências suficientes → retorne o FORMATO B.

A nota deve seguir estritamente a rubrica fornecida.
Se as evidências forem poucas, mencione menor confiança na justificativa (campo lacunas).

REGRA CRÍTICA: Ausência de evidência ≠ desempenho ruim.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO A — com nota:
{{
  "criterio": "{nome_c}",
  "nota": 3,
  "peso": {peso_c},
  "contribuicao": {round(3 * peso_c / 5, 1)},
  "evidencias": [
    {{
      "trecho": "[N] 'trecho da evidência positiva usada'",
      "interpretacao": "Uma frase sobre o que esse trecho demonstra."
    }}
  ],
  "lacunas": "O que as evidências negativas revelam como ausente ou inconsistente."
}}

FORMATO B — sem evidência suficiente:
{{
  "criterio": "{nome_c}",
  "peso": {peso_c},
  "sem_evidencia": true,
  "motivo": "Explicação de por que as evidências não permitem avaliar este critério.",
  "evidencia_esperada": "Que tipo de sinal seria necessário para avaliar este critério."
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
        # Server-side guard: demote to sem_evidencia if evidence count is below minimum
        dados = _aplicar_minimo_evidencias(dados)

    return dados


def avaliar_candidato(scorecard: dict, nome_candidato: str, transcricao: str) -> dict:
    """Evaluate a candidate against a full scorecard using a two-stage process.

    Stage 1: extract positive and negative evidence for each criterion from the transcript.
    Stage 2: score each criterion using only the evidence from Stage 1.
    """
    criterios = scorecard["criterios"]

    # Stage 1 — extract evidence per criterion (positive and negative)
    ev_por_criterio = _extrair_evidencias_por_criterio(criterios, transcricao, nome_candidato)

    # Build per-criterion blocks for the scoring prompt
    blocos = []
    for c in criterios:
        nome = c["nome"]
        peso = c["peso"]
        rubrica = "\n".join(f"  {k}: {v}" for k, v in sorted(c["rubrica"].items()))
        ev = ev_por_criterio.get(nome, {})
        positivas = ev.get("positivas", [])
        negativas = ev.get("negativas", [])

        pos_txt = (
            "\n".join(f"  + {e['trecho']} — {e['interpretacao']}" for e in positivas)
            or "  (nenhuma)"
        )
        neg_txt = (
            "\n".join(f"  - {e['trecho']} — {e['interpretacao']}" for e in negativas)
            or "  (nenhuma)"
        )

        blocos.append(
            f"CRITÉRIO: {nome} (Peso: {peso}%)\n"
            f"Rubrica:\n{rubrica}\n"
            f"Evidências positivas ({len(positivas)}):\n{pos_txt}\n"
            f"Evidências negativas ({len(negativas)}):\n{neg_txt}"
        )

    blocos_txt = "\n\n".join(blocos)

    # Stage 2 — score all criteria using only the extracted evidence
    prompt = f"""Você é um entrevistador especialista em avaliação estruturada de candidatos.
Avalie o candidato abaixo usando APENAS as evidências fornecidas para cada critério.
Não busque novas evidências — use somente o que está listado.

Candidato: {nome_candidato}

{blocos_txt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES — LEIA COM ATENÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Para CADA critério acima:
1. Use APENAS as evidências listadas para decidir a nota — não busque novas na transcrição
2. SE houver evidências suficientes → avalie com nota 1–5 conforme a rubrica
3. SE não houver evidências suficientes → retorne o objeto sem_evidencia
4. A nota deve seguir estritamente a rubrica fornecida
5. Se as evidências forem poucas, mencione menor confiança no campo lacunas

REGRA CRÍTICA: Ausência de evidência NÃO significa desempenho ruim.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATOS DOS OBJETOS DE AVALIAÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Objeto COM nota (quando há evidências para avaliar):
{{
  "criterio": "NomeDoCriterio",
  "nota": 3,
  "peso": 20,
  "contribuicao": 12.0,
  "evidencias": [
    {{
      "trecho": "[N] 'trecho da evidência positiva usada'",
      "interpretacao": "Uma frase sobre o que esse trecho demonstra."
    }}
  ],
  "lacunas": "O que as evidências negativas revelam como ausente; inclua nota de baixa confiança se poucas evidências."
}}

Objeto SEM evidência suficiente:
{{
  "criterio": "NomeDoCriterio",
  "peso": 20,
  "sem_evidencia": true,
  "motivo": "Explicação de por que as evidências não permitem avaliar este critério.",
  "evidencia_esperada": "Que tipo de sinal seria necessário para avaliar este critério."
}}

Regras adicionais para objetos COM nota:
- nota: inteiro de 1 a 5, estritamente conforme a rubrica
- contribuicao = (nota × peso) / 5
- evidencias: use os trechos das evidências positivas relevantes
- lacunas: o que ficou ausente ou inconsistente conforme as evidências negativas

Retorne APENAS JSON válido (sem markdown, sem explicação):
{{
  "avaliacoes": [
    {{ objeto com nota OU sem evidência para cada critério }}
  ]
}}"""

    texto = _chamar_api(prompt, max_tokens=4096)

    try:
        bruto = _extrair_json(texto)
        dados = json.loads(bruto)
    except json.JSONDecodeError as e:
        print(f"[AI PARSING ERROR] Resposta bruta da avaliação:\n{texto}")
        raise AIParsingError(f"A IA retornou JSON inválido na avaliação: {e}") from e

    # Server-side: recalculate contributions, sort evidence, enforce minimum evidence rule
    avaliacoes_processadas = []
    for av in dados["avaliacoes"]:
        if av.get("sem_evidencia"):
            av.pop("nota", None)
            av.pop("contribuicao", None)
            av.pop("evidencias", None)
            av.pop("lacunas", None)
            avaliacoes_processadas.append(av)
        else:
            av["contribuicao"] = round((av["nota"] * av["peso"]) / 5, 1)
            av["evidencias"] = _ordenar_evidencias(av.get("evidencias") or [])
            # Demote to sem_evidencia if evidence count is below the required minimum
            avaliacoes_processadas.append(_aplicar_minimo_evidencias(av))
    dados["avaliacoes"] = avaliacoes_processadas

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
