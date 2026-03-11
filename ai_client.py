"""
AI client for Cabine — Anthropic Claude integration.

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

# Sentence-transformers for semantic similarity in transcript segmentation.
# Optional: if not installed, segmentation falls back to lexical-only mode.
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    import numpy as _np
    _EMBEDDINGS_DISPONIVEL = True
except ImportError:
    _EMBEDDINGS_DISPONIVEL = False

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


# ── Fixed technical format specs (never shown in the prompt editor) ────────────
# Each function appends its format block to the human-editable prompt from the DB.
# Use {{ / }} for literal braces (these strings go through .format()).

_FORMAT_GERAR_SCORECARD = """\
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
{
  "criterios": [
    {
      "nome": "UmaPalavra",
      "peso": 20,
      "rubrica": {
        "1": "Sinal observável na resposta — nível 1 (máx 200 caracteres)",
        "2": "Sinal observável na resposta — nível 2 (máx 200 caracteres)",
        "3": "Sinal observável na resposta — nível 3 (máx 200 caracteres)",
        "4": "Sinal observável na resposta — nível 4 (máx 200 caracteres)",
        "5": "Sinal observável na resposta — nível 5 (máx 200 caracteres)"
      }
    }
  ]
}"""

_FORMAT_REGENERAR_CRITERIO = """\
FORMATO DE SAÍDA:
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

_FORMAT_GERAR_RUBRICA = """\
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

_FORMAT_AVALIAR_CANDIDATO = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES DE AVALIAÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Antes de atribuir qualquer nota: leia TODAS as evidências relevantes da transcrição e só então decida a nota.
Interprete a rubrica literalmente — use os descritores de cada nível exatamente como escritos.

USO DAS TRANSCRIÇÕES:
- Transcrição legível (em blocos): gramática corrigida — use para analisar o CONTEÚDO das respostas.
- Transcrição original numerada: fala real do candidato — use adicionalmente para critérios que avaliam
  COMUNICAÇÃO, clareza ou organização verbal. Observe: hesitações, repetições, dificuldade de estruturar
  frases, respostas confusas ou desorganizadas (esses sinais somem na versão legível).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGRAS PARA EVIDÊNCIAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A transcrição está organizada em blocos de contexto — avalie sempre o raciocínio completo do candidato
dentro do bloco, nunca com base em frases isoladas.
- Cite o intervalo de linhas que contém a evidência. O sistema exibirá o trecho completo automaticamente.
  Linha única:  "[12]"
  Intervalo:    "[12-14]"
- SEMPRE prefira intervalos que abranjam a resposta completa do candidato — evite citar uma única linha
  quando a resposta se estende por várias linhas. Exemplo: use "[52-55]" em vez de "[53]".
- Nunca cite uma linha cujo significado dependa das linhas ao redor sem incluí-las no intervalo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATOS DOS OBJETOS DE AVALIAÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Objeto COM nota (quando há evidência para avaliar):
{
  "criterio": "NomeDoCriterio",
  "nota": 3,
  "peso": 20,
  "contribuicao": 12.0,
  "evidencias": ["[12-14]"],
  "lacunas": "O que não foi demonstrado"
}

Objeto SEM evidência (quando a transcrição não permite avaliar este critério):
{
  "criterio": "NomeDoCriterio",
  "peso": 20,
  "sem_evidencia": true,
  "motivo": "Explicação breve de por que a transcrição não permite avaliar este critério.",
  "evidencia_esperada": "Que tipo de falas ou situações seriam necessárias para avaliar este critério."
}

Regras adicionais para objetos COM nota:
- nota: inteiro de 1 a 5, estritamente de acordo com a rubrica
- contribuicao = (nota × peso) / 5
- evidencias: intervalo de linhas — linha única "[N]" ou intervalo "[N-M]" (sem texto adicional)
- lacunas: o que ficou ausente ou não foi demonstrado

Retorne APENAS JSON válido (sem markdown, sem explicação):
{
  "avaliacoes": [
    { objeto com nota OU sem evidência para cada critério }
  ]
}"""


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


# ── Transcript block segmentation ──────────────────────────────────────────────

# Speaker label prefixes recognised in Portuguese transcripts
_ENTREVISTADOR_PREFIXOS = ("entrevistador:", "entrevistadora:", "e:", "interviewer:", "i:")
_CANDIDATO_PREFIXOS = ("candidato:", "candidata:", "c:", "candidate:", "respondente:", "r:")

# Block size limits (in parsed lines, blank lines excluded)
_BLOCO_MIN_LINHAS = 2
_BLOCO_MAX_LINHAS = 15

# ── Hybrid topic-change thresholds ─────────────────────────────────────────────
# When sentence-transformers is available, the final score is:
#   score = _PESO_SEMANTICO * cos_sim + _PESO_LEXICAL * jaccard + bonuses
# Below _LIMIAR_TOPICO_COMBINADO the question is treated as a new topic.
# When embeddings are unavailable the system falls back to lexical-only and uses
# the original _LIMIAR_TOPICO threshold.
_LIMIAR_TOPICO_COMBINADO = 0.35   # hybrid threshold (embedding + lexical)
_LIMIAR_TOPICO = 0.07             # lexical-only fallback threshold
_PESO_SEMANTICO = 0.85            # weight of cosine similarity in combined score
_PESO_LEXICAL = 0.15              # weight of Jaccard similarity in combined score

# How many of the most recent block lines to use for the block embedding.
# Using a sliding window keeps the representation focused on the active thread.
_JANELA_BLOCO_EMBEDDING = 8

# Score bonuses that make the system more conservative (prefer not splitting):
_BONUS_NARRATIVO = 0.08    # question starts with a narrative continuation pattern
_BONUS_BLOCO_PEQUENO = 0.05  # current block is still small (< 5 lines)
_BONUS_REFERENCIA = 0.10   # question contains a demonstrative reference pronoun

# Minimum number of content keywords in the new question to trigger a topic comparison.
# Questions with fewer keywords are almost certainly short probes — keep in block.
_MIN_KEYWORDS_PERGUNTA = 3

# Portuguese suffix list for light stemming, longest-first so the most specific
# suffix is stripped first (e.g. "amentos" before "os" before "s").
_SUFIXOS_STEM = (
    "amentos", "imentos", "imento",
    "ações", "anças", "agens", "entes", "mente",
    "ação", "ança", "agem", "uras", "osas", "osos",
    "ivas", "ivos", "adas", "ados", "idas", "idos", "ente",
    "ando", "endo", "avam",
    "ura", "osa", "oso", "iva", "ivo", "ada", "ado", "ida", "ido",
    "ias", "ava", "ões",
    "ia", "ar", "er", "ir", "ou", "iu",
    "es", "os", "as",
    "s",
)

# Syntactic patterns that strongly suggest the interviewer is probing deeper
# into the same topic rather than introducing a new one.
_PROBE_NARRATIVO_RE = re.compile(
    r"^(?:"
    r"e\s+(?:depois|então|daí|como|o\s+que)\b|"   # E depois / E então / E como
    r"como\s+você\s+\w|"                           # Como você [verbo]...
    r"por\s+que\s+(?:você|isso)\b|"                # Por que você / Por que isso
    r"o\s+que\s+(?:você|aconteceu|deu|havia)\b|"   # O que você / O que aconteceu
    r"qual\s+foi\s+|"                              # Qual foi...
    r"quais\s+foram\s+"                            # Quais foram...
    r")",
    re.IGNORECASE,
)

# Demonstrative pronouns that indicate the question refers back to something
# already mentioned in the current block ("isso", "esse problema", etc.).
# Presence of these is a strong signal that the question continues the same topic.
_PRONOMES_REFERENCIA_RE = re.compile(
    r"\b(?:isso|esse|essa|este|esta|aquilo|nesse|nessa|neste|nesta|aquele|aquela)\b",
    re.IGNORECASE,
)

# Words filtered out before keyword comparison.
# Includes both general Portuguese function words and high-frequency interview
# domain words that appear in almost every question/answer and carry no topic signal.
_STOPWORDS_PT: frozenset[str] = frozenset({
    # Function words
    "a", "o", "e", "é", "de", "da", "do", "das", "dos", "em", "na", "no",
    "nas", "nos", "um", "uma", "uns", "umas", "para", "por", "com", "não",
    "se", "que", "mas", "ou", "ao", "à", "como", "mais", "me", "te",
    "lhe", "já", "bem", "assim", "isso", "este", "esta", "esse", "essa",
    "aquele", "aquela", "aqui", "ali", "lá", "aí",
    "seu", "sua", "seus", "suas", "meu", "minha", "meus", "minhas",
    "nosso", "nossa", "nossos", "nossas",
    "você", "ele", "ela", "eles", "elas", "nós", "vocês",
    "foi", "ser", "ter", "há", "quando", "onde", "porque", "então", "sobre",
    "também", "depois", "antes", "até", "desde", "durante", "sim",
    "muito", "pouco", "todo", "toda", "todos", "todas", "cada",
    "qual", "quais", "quem", "quanto", "tudo", "nada", "algo",
    "alguém", "ninguém", "agora", "sempre", "nunca",
    "ainda", "apenas", "só", "mesmo", "próprio", "outro", "outra",
    "outros", "outras", "novo", "nova", "novos", "novas",
    # Generic interview-domain words that appear in virtually every question/answer
    # and therefore add noise rather than signal to topic similarity
    "fale", "conte", "contou", "contar", "descreva", "explique", "explica",
    "diga", "mencione", "descrever", "gostaria", "quero", "queremos",
    "entender", "entenda", "compreender", "conhecer",
    "situação", "situações", "exemplo", "exemplos",
    "vez", "vezes", "momento", "momentos",
    "contexto", "experiência", "experiências",
    "caso", "casos", "história", "histórias",
    "pode", "poderia", "consegue", "conseguiu",
    "aconteceu", "acontece", "aconteceu", "ocorreu", "ocorre",
})


def _detectar_falante(linha: str) -> tuple[str, str]:
    """Return (speaker_type, text_without_prefix).

    speaker_type is 'entrevistador', 'candidato', or 'unknown'.
    """
    lower = linha.lower()
    for p in _ENTREVISTADOR_PREFIXOS:
        if lower.startswith(p):
            return "entrevistador", linha[len(p):].strip()
    for p in _CANDIDATO_PREFIXOS:
        if lower.startswith(p):
            return "candidato", linha[len(p):].strip()
    return "unknown", linha


def _extrair_palavras_chave(texto: str) -> frozenset[str]:
    """Extract content words (≥ 3 chars, not stopwords) from a text."""
    tokens = re.findall(r"\b[a-záéíóúâêîôûãõàèìòùç]{3,}\b", texto.lower())
    return frozenset(t for t in tokens if t not in _STOPWORDS_PT)


def _radical_pt(palavra: str) -> str:
    """Light Portuguese stemmer: strips the longest known suffix requiring ≥ 4-char stem.

    Handles the most common morphological patterns in interview transcripts:
    conjugations (liderou/liderava → lider), nominalizations (liderança → lider),
    plurals (conflitos → conflito), participles (resolvido → resolv), etc.
    """
    for sfx in _SUFIXOS_STEM:
        if palavra.endswith(sfx) and len(palavra) - len(sfx) >= 4:
            return palavra[:-len(sfx)]
    return palavra


def _peso_palavra(stem: str) -> float:
    """Return a specificity weight for a stem.

    Longer stems are more topic-specific and contribute more to similarity.
    """
    n = len(stem)
    if n >= 10:
        return 3.0
    if n >= 8:
        return 2.0
    if n >= 6:
        return 1.5
    return 1.0


def _similaridade_semantica(kw_a: frozenset[str], kw_b: frozenset[str]) -> float:
    """Weighted stem-aware Jaccard similarity between two keyword sets.

    Steps:
    1. Stem every keyword in both sets.
    2. Compute the weighted intersection / weighted union where each stem's
       weight is proportional to its length (longer = more specific).

    This handles morphological variants (liderou/liderança → lider) so that
    topically related questions that use different word forms still score well.
    """
    if not kw_a or not kw_b:
        return 0.0

    stems_a = {_radical_pt(w) for w in kw_a}
    stems_b = {_radical_pt(w) for w in kw_b}

    intersecao = stems_a & stems_b
    uniao = stems_a | stems_b

    if not uniao:
        return 0.0

    peso_intersecao = sum(_peso_palavra(s) for s in intersecao)
    peso_uniao = sum(_peso_palavra(s) for s in uniao)
    return peso_intersecao / peso_uniao


def _e_sinal_continuacao_narrativa(pergunta: str) -> bool:
    """Return True if the question starts with a narrative continuation pattern.

    These openers ("Como você…", "Por que você…", "O que aconteceu…", etc.)
    indicate the interviewer is probing deeper into the current story rather
    than introducing a new topic, regardless of keyword overlap.
    """
    return bool(_PROBE_NARRATIVO_RE.match(pergunta.strip()))


def _tem_referencia_proxima(pergunta: str) -> bool:
    """Return True if the question contains a demonstrative reference pronoun.

    Words like "isso", "esse problema", "essa situação" strongly indicate the
    question refers to something already mentioned in the current block.
    """
    return bool(_PRONOMES_REFERENCIA_RE.search(pergunta))


# ── Embedding helpers (sentence-transformers) ───────────────────────────────────

_modelo_embeddings = None  # lazy-loaded singleton


def _get_modelo_embeddings():
    """Return the (lazily-loaded) sentence-transformer model.

    The model is downloaded on first use (~118 MB) and kept in memory for the
    lifetime of the process.  Raises ImportError if sentence-transformers is
    not installed.
    """
    global _modelo_embeddings
    if _modelo_embeddings is None:
        _modelo_embeddings = _SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _modelo_embeddings


def _calcular_embedding(texto: str):
    """Return a normalised embedding vector for *texto*."""
    return _get_modelo_embeddings().encode(texto, normalize_embeddings=True)


def _similaridade_cosseno(a, b) -> float:
    """Cosine similarity between two pre-normalised embedding vectors.

    Since both vectors are L2-normalised, cosine similarity equals the dot product.
    Result is in [−1, 1]; for semantically similar sentences in practice [0.1, 0.9].
    """
    return float(_np.dot(a, b))


def _e_novo_topico(nova_pergunta: str, bloco_atual: list[dict]) -> bool:
    """Return True if the interviewer's new question introduces a different topic.

    Decision logic
    ─────────────
    1. Short-probe fast path: questions with fewer than _MIN_KEYWORDS_PERGUNTA
       content words ("Como assim?", "E daí?") are almost certainly follow-ups —
       keep in block unconditionally.

    2. Accumulate score bonuses from heuristic signals (all signals that bias
       toward keeping the block together):
       • Narrative opener  (+_BONUS_NARRATIVO)   — "Como você…", "Por que…", etc.
       • Small block       (+_BONUS_BLOCO_PEQUENO) — block has fewer than 5 lines
       • Demonstrative ref (+_BONUS_REFERENCIA)   — contains "isso", "esse X", etc.

    3a. Hybrid path (sentence-transformers available):
        • Compute cosine similarity between the new question and the most recent
          _JANELA_BLOCO_EMBEDDING lines of the current block (embedding-based).
        • Compute weighted Jaccard on stemmed keywords (lexical-based).
        • Final score = 0.85 × cos_sim + 0.15 × jaccard + bonuses
        • Threshold: _LIMIAR_TOPICO_COMBINADO (0.35)

    3b. Lexical-only fallback (no sentence-transformers):
        • Final score = jaccard + bonuses
        • Threshold: _LIMIAR_TOPICO (0.07)

    In both cases: score ≥ threshold → same topic (keep); below → new block.
    When in doubt the system is conservative and prefers not to split.
    """
    kw_pergunta = _extrair_palavras_chave(nova_pergunta)
    if len(kw_pergunta) < _MIN_KEYWORDS_PERGUNTA:
        return False  # short probe — keep in block unconditionally

    # ── Heuristic bonuses ───────────────────────────────────────────────────────
    bonus = 0.0
    if _e_sinal_continuacao_narrativa(nova_pergunta):
        bonus += _BONUS_NARRATIVO
    if len(bloco_atual) < 5:
        bonus += _BONUS_BLOCO_PEQUENO
    if _tem_referencia_proxima(nova_pergunta):
        bonus += _BONUS_REFERENCIA

    # Use the most recent lines of the block for a focused representation.
    janela = bloco_atual[-_JANELA_BLOCO_EMBEDDING:]
    texto_bloco = " ".join(item["texto"] for item in janela)

    # ── Lexical similarity (always computed) ────────────────────────────────────
    kw_bloco = _extrair_palavras_chave(texto_bloco)
    sim_lexical = _similaridade_semantica(kw_pergunta, kw_bloco)

    # ── Hybrid path: embeddings + lexical ──────────────────────────────────────
    if _EMBEDDINGS_DISPONIVEL:
        try:
            emb_pergunta = _calcular_embedding(nova_pergunta)
            emb_bloco = _calcular_embedding(texto_bloco)
            sim_semantica = _similaridade_cosseno(emb_pergunta, emb_bloco)
        except Exception:
            sim_semantica = 0.0  # embedding failed — degrade gracefully

        score = _PESO_SEMANTICO * sim_semantica + _PESO_LEXICAL * sim_lexical + bonus
        return score < _LIMIAR_TOPICO_COMBINADO

    # ── Lexical-only fallback ───────────────────────────────────────────────────
    score = sim_lexical + bonus
    return score < _LIMIAR_TOPICO


def _segmentar_em_blocos(transcricao: str) -> str:
    """Reorganise a transcript into semantic context blocks.

    Each block groups a main question + the candidate's complete answer +
    follow-up probes that are topically related to the same idea.
    A new block begins only when the interviewer's question introduces a
    clearly different topic (detected via keyword similarity) or when the
    current block has grown beyond _BLOCO_MAX_LINHAS.

    Line numbers from the original transcript are preserved so the AI can
    still cite specific lines as evidence.

    If the transcript has no recognisable speaker labels (e.g. raw Whisper
    output), paragraphs separated by blank lines are used as blocks; if
    there are no blank lines either, lines are grouped in chunks of 8.
    """
    linhas_raw = transcricao.strip().split("\n")

    # Parse every non-blank line
    parsed: list[dict] = []
    for i, linha in enumerate(linhas_raw):
        stripped = linha.strip()
        if stripped:
            falante, texto = _detectar_falante(stripped)
            parsed.append({"num": i + 1, "falante": falante, "texto": texto, "original": stripped})

    if not parsed:
        return transcricao

    tem_labels = any(p["falante"] != "unknown" for p in parsed)

    if tem_labels:
        blocks: list[list[dict]] = []
        current: list[dict] = [parsed[0]]

        for item in parsed[1:]:
            if item["falante"] == "entrevistador":
                # Split if: block is at max size OR the question changes topic
                if len(current) >= _BLOCO_MAX_LINHAS or _e_novo_topico(item["texto"], current):
                    blocks.append(current)
                    current = [item]
                else:
                    current.append(item)
            else:
                current.append(item)

        if current:
            blocks.append(current)
    else:
        # Fallback: split on blank lines
        blocks = []
        current_block_items: list[dict] = []
        parsed_idx = 0
        for raw in linhas_raw:
            if raw.strip() == "":
                if current_block_items:
                    blocks.append(current_block_items)
                    current_block_items = []
            else:
                if parsed_idx < len(parsed):
                    current_block_items.append(parsed[parsed_idx])
                    parsed_idx += 1
        if current_block_items:
            blocks.append(current_block_items)

        # No blank lines found → group every 8 lines
        if len(blocks) <= 1 and len(parsed) > 8:
            blocks = [parsed[i:i + 8] for i in range(0, len(parsed), 8)]

    # Merge blocks that are too short into the previous one
    merged: list[list[dict]] = []
    for block in blocks:
        if merged and len(block) < _BLOCO_MIN_LINHAS:
            merged[-1].extend(block)
        else:
            merged.append(block)

    # Format output with block headers and preserved line numbers
    partes: list[str] = []
    for b_idx, block in enumerate(merged, start=1):
        inicio = block[0]["num"]
        fim = block[-1]["num"]
        header = f"━━ Bloco {b_idx} (linhas {inicio}–{fim}) ━━"
        linhas_bloco = [f"[{item['num']}] {item['original']}" for item in block]
        partes.append(header + "\n" + "\n".join(linhas_bloco))

    return "\n\n".join(partes)


def _ordenar_evidencias(evidencias: list[str]) -> list[str]:
    """Sort evidence citations by their transcript line number.

    Supports both single-line format  [N]
    and range format                  [N-M].
    """
    def _linha(ev: str) -> int:
        m = re.match(r"\[(\d+)", ev.strip())
        return int(m.group(1)) if m else 999999
    return sorted(evidencias, key=_linha)


_LEGIVEL_CHUNK_SIZE = 150  # max lines per API call when normalising long transcripts

_LEGIVEL_PROMPT_TEMPLATE = """\
Você é um assistente de normalização de transcrições de entrevistas.

TAREFA: Criar uma versão legível do trecho de transcrição abaixo, corrigindo apenas erros de escrita.

REGRAS OBRIGATÓRIAS:
1. Mantenha EXATAMENTE o mesmo número de linhas — NUNCA junte nem divida linhas.
2. Mantenha todos os números de linha [N] e prefixos de falante (entrevistador:, candidato:, etc.) exatamente como estão.
3. Pode corrigir: ortografia, concordância gramatical básica, pontuação, capitalização do início de frases.
4. PROIBIDO: remover palavras, apagar hesitações ("tipo", "ahn", "né", "é"), resumir, reorganizar ideias, ou reescrever frases completamente.
5. Linhas em branco [N] devem permanecer em branco [N] — nunca adicione texto a elas.
6. Retorne APENAS as linhas numeradas, sem explicações, comentários ou formatação extra.

EXEMPLO:
Entrada:  [3] candidato: eu eu acho que tipo dados sao muito importante pra decisao
Saída:    [3] candidato: Eu eu acho que tipo dados são muito importantes para a decisão.

TRECHO ORIGINAL:
{linhas_numeradas}

VERSÃO LEGÍVEL:"""


def _normalizar_chunk(linhas_chunk: list[str], offset: int) -> dict[int, str]:
    """Normalise a slice of transcript lines; return a {line_num: corrected_text} map.

    *offset* is the 0-based index of the first line in the full transcript so
    that line numbers in the prompt match the originals.  On any API failure or
    low coverage the chunk falls back to its original text (guaranteeing that
    callers always get full coverage).
    """
    linhas_numeradas = "\n".join(
        f"[{offset + j + 1}] {linha}" if linha.strip() else f"[{offset + j + 1}]"
        for j, linha in enumerate(linhas_chunk)
    )
    prompt = _LEGIVEL_PROMPT_TEMPLATE.format(linhas_numeradas=linhas_numeradas)
    max_tokens = min(8192, max(2048, sum(len(l) for l in linhas_chunk) // 3 + 512))

    # Valid line numbers for this chunk — used to reject hallucinated line numbers
    linhas_validas = {offset + j + 1 for j in range(len(linhas_chunk))}

    mapa: dict[int, str] = {}
    try:
        resultado = _chamar_api(prompt, max_tokens=max_tokens)
        for linha_res in resultado.strip().split("\n"):
            m = re.match(r"\[(\d+)\]\s*(.*)", linha_res)
            if m:
                n = int(m.group(1))
                if n in linhas_validas:  # reject any line number not in the original chunk
                    mapa[n] = m.group(2)
    except Exception:
        pass  # fall through to fallback

    # Structural validation: any content line absent from the map reverts to original
    for j, linha in enumerate(linhas_chunk):
        n = offset + j + 1
        if linha.strip() and n not in mapa:
            mapa[n] = linha  # line-level fallback preserves structure

    return mapa


def gerar_transcricao_legivel(transcricao_original: str) -> str:
    """Generate a normalised readable version of the transcript for AI comprehension.

    Allowed corrections: orthography, basic grammar, punctuation, capitalisation.
    NOT allowed: removing words, removing hesitations ("tipo", "ahn", "né"),
    merging or splitting lines, summarising, or changing meaning.

    The line count is strictly preserved so that line numbers in the legible
    version map 1-to-1 to the original transcript.

    Long transcripts are processed in chunks of _LEGIVEL_CHUNK_SIZE lines so
    that the API call stays within output-token limits.  Each chunk falls back
    to the original text if the API call fails or returns incomplete output.

    Skipped for very short (< 5 lines) transcripts.
    """
    linhas = transcricao_original.strip().split("\n")
    n_linhas = len(linhas)

    if n_linhas < 5:
        return transcricao_original

    # Process in chunks (single chunk when transcript is short enough)
    mapa: dict[int, str] = {}
    for start in range(0, n_linhas, _LEGIVEL_CHUNK_SIZE):
        chunk = linhas[start:start + _LEGIVEL_CHUNK_SIZE]
        mapa.update(_normalizar_chunk(chunk, start))

    # Reassemble preserving blank line positions
    saida: list[str] = []
    for i, linha in enumerate(linhas):
        n = i + 1
        if not linha.strip():
            saida.append("")
        elif n in mapa:
            saida.append(mapa[n])
        else:
            saida.append(linha)  # safety fallback (should never be reached)

    return "\n".join(saida)


def resolver_evidencia(ev: str, transcricao: str) -> "dict | None":
    """Parse [N] or [N-M] from an evidence string and return the actual transcript lines.

    Handles both the new format ("[12-14]") and the legacy format
    ("[12-14] 'inline quote'") so stored evaluations continue to work.

    Returns:
        {"ref": "[12–14]", "linhas": ["line 12 text", "line 13 text", ...]}
        or None if the reference cannot be parsed or no lines are found.
    """
    m = re.match(r"\[(\d+)(?:-(\d+))?\]", ev.strip())
    if not m:
        return None
    inicio = int(m.group(1))
    fim = int(m.group(2)) if m.group(2) else inicio

    linhas_raw = transcricao.strip().split("\n")
    trecho = []
    for n in range(inicio, fim + 1):
        if 1 <= n <= len(linhas_raw):
            linha = linhas_raw[n - 1].strip()
            if linha:
                trecho.append(linha)

    if not trecho:
        return None

    ref = f"[{inicio}]" if inicio == fim else f"[{inicio}–{fim}]"
    return {"ref": ref, "linhas": trecho}


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


def _chamar_api(prompt: str, max_tokens: int, temperature: float = 1.0) -> str:
    """
    Call the Anthropic API and return the text of the first content block.
    Translates SDK exceptions to our AIError hierarchy.
    """
    try:
        client = _get_client()
        resposta = client.messages.create(
            model=MODELO,
            max_tokens=max_tokens,
            temperature=temperature,
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

def gerar_scorecard(nome_cargo: str, descricao_vaga: str, prompt_template: str | None = None) -> dict:
    if prompt_template is not None:
        contexto = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"CONTEXTO\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Cargo: {nome_cargo}\n"
            f"Descrição da vaga:\n{descricao_vaga}"
        )
        prompt = contexto + "\n\n" + prompt_template + "\n\n" + _FORMAT_GERAR_SCORECARD
    else:
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
    prompt_template: str | None = None,
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
    outros_nomes_str = ', '.join(outros_nomes)

    if prompt_template is not None:
        contexto = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"CONTEXTO\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Cargo: {nome_cargo}\n"
            f"Descrição da vaga:\n{descricao_vaga}\n\n"
            f"Critério anterior a ser substituído: \"{criterio_anterior}\"\n"
            f"Outros critérios já definidos no scorecard (não repita nenhum): {outros_nomes_str}"
        )
        format_block = _FORMAT_REGENERAR_CRITERIO.format(max_palavras=max_palavras)
        prompt = contexto + "\n\n" + prompt_template + "\n\n" + format_block
    else:
        prompt = f"""Você é um especialista em People & Culture com foco em avaliação comportamental estruturada.
Você está atualizando um scorecard de entrevista. Precisa gerar UM NOVO critério comportamental para substituir um existente.

Cargo: {nome_cargo}
Descrição da Vaga:
{descricao_vaga}

CRITÉRIO ANTERIOR A SER SUBSTITUÍDO: "{criterio_anterior}"
IMPORTANTE: O novo critério deve ser DIFERENTE e NÃO PARECIDO com "{criterio_anterior}".
Deve representar um conceito comportamental distinto — não o mesmo tema com outras palavras.

Os outros 4 critérios já definidos no scorecard (NÃO repita nenhum destes nem o critério anterior):
{outros_nomes_str}

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
    prompt_template: str | None = None,
) -> dict:
    """Generate rubric levels 1-5 for a manually-named criterion.

    max_palavras: maximum word count allowed per description level.
    Returns a rubric dict with keys "1" through "5".
    Raises AIParsingError if the response is invalid.
    """
    outros_str = ", ".join(outros_criterios) if outros_criterios else "N/A"

    if prompt_template is not None:
        contexto = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"CONTEXTO\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Cargo: {nome_cargo}\n"
            f"Descrição da vaga:\n{descricao_vaga}\n\n"
            f"Critério a descrever: {nome_criterio}\n"
            f"Outros critérios do scorecard para contexto: {outros_str}"
        )
        format_block = _FORMAT_GERAR_RUBRICA.format(max_palavras=max_palavras)
        prompt = contexto + "\n\n" + prompt_template + "\n\n" + format_block
    else:
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


def avaliar_criterio_unico(
    criterio: dict,
    nome_candidato: str,
    transcricao: str,
    transcricao_legivel: "str | None" = None,
) -> dict:
    """Evaluate a single criterion for a candidate given their transcript.

    transcricao_legivel: pre-generated readable version of the transcript.
        If provided, generation is skipped (pass the stored value to ensure
        consistency across re-evaluations).  When None, it is generated here.

    Returns one avaliacao dict. Two possible shapes:
    - Scored:   {"criterio", "nota", "peso", "contribuicao", "evidencias", "lacunas"}
    - Unscored: {"criterio", "peso", "sem_evidencia": True, "motivo", "evidencia_esperada"}
    Raises AIParsingError if the response is invalid.
    """
    # ── Preprocessing: use provided legible version or generate one ─────────────
    if transcricao_legivel is None:
        transcricao_legivel = gerar_transcricao_legivel(transcricao)
    blocos = _segmentar_em_blocos(transcricao_legivel)
    original_numerada = _numerar_linhas(transcricao)

    rubrica_txt = "\n".join(f"  {k}: {v}" for k, v in sorted(criterio["rubrica"].items()))
    nome_c = criterio["nome"]
    peso_c = criterio["peso"]

    prompt = f"""Você é um entrevistador especialista em avaliação estruturada de candidatos.
Avalie o candidato abaixo para UM ÚNICO critério comportamental.

Candidato: {nome_candidato}

CRITÉRIO: {nome_c} (Peso: {peso_c}%)
Rubrica:
{rubrica_txt}

=== TRANSCRIÇÃO LEGÍVEL (organizada em blocos de contexto) ===
Use para analisar o CONTEÚDO das respostas (raciocínio, exemplos, profundidade).
Cada bloco agrupa uma pergunta principal + a resposta completa do candidato + perguntas de aprofundamento relacionadas.
Os números de linha permitem citar trechos como evidência.
{blocos}

=== TRANSCRIÇÃO ORIGINAL (fala real do candidato) ===
Use adicionalmente se este critério avaliar COMUNICAÇÃO, clareza ou organização verbal.
Observe: hesitações, repetições, dificuldade de estruturar frases, respostas confusas ou desorganizadas.
{original_numerada}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Antes de atribuir nota: leia TODAS as evidências relevantes da transcrição e só então decida a nota.
Interprete a rubrica literalmente — use os descritores de cada nível exatamente como escritos.

PASSO 1 — Verifique se há evidência real na transcrição para este critério.
Leia os blocos por completo e procure falas do candidato que demonstrem diretamente o comportamento descrito na rubrica.

SE houver evidência suficiente → avalie com nota 1–5 e retorne o formato A.
SE NÃO houver evidência suficiente → NÃO atribua nota e retorne o formato B.

REGRA CRÍTICA: Ausência de evidência ≠ desempenho ruim.
Não invente contexto. Não assuma coisas não ditas na transcrição.

REGRAS PARA EVIDÊNCIAS:
- Avalie sempre o raciocínio completo do candidato no contexto do bloco — nunca com base em frases isoladas.
- Cite o intervalo de linhas que contém a evidência. O sistema exibirá o trecho completo automaticamente.
  Formato de linha única:  "[12]"
  Formato de intervalo:    "[12-14]"
- SEMPRE prefira intervalos que abranjam a resposta completa — evite citar uma única linha quando a
  resposta se estende por várias linhas. Exemplo: use "[52-55]" em vez de "[53]".
- Nunca cite uma linha cujo significado dependa das linhas ao redor sem incluí-las no intervalo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO A — com evidência (use quando há base na transcrição para avaliar):
{{
  "criterio": "{nome_c}",
  "nota": 3,
  "peso": {peso_c},
  "contribuicao": {round(3 * peso_c / 5, 1)},
  "evidencias": ["[12-14]"],
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

    texto = _chamar_api(prompt, max_tokens=1024, temperature=0)

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


def avaliar_candidato(
    scorecard: dict,
    nome_candidato: str,
    transcricao: str,
    prompt_template: "str | None" = None,
    transcricao_legivel: "str | None" = None,
) -> dict:
    """Evaluate a candidate against a scorecard.

    transcricao_legivel: pre-generated readable version of the transcript.
        Pass the stored value to guarantee evaluation consistency across
        re-runs of the same interview.  When None, it is generated here.
    """
    # ── Preprocessing: use provided legible version or generate one ─────────────
    if transcricao_legivel is None:
        transcricao_legivel = gerar_transcricao_legivel(transcricao)
    blocos = _segmentar_em_blocos(transcricao_legivel)
    original_numerada = _numerar_linhas(transcricao)

    blocos_criterio = []
    for c in scorecard["criterios"]:
        rubrica = "\n".join(f"  {k}: {v}" for k, v in sorted(c["rubrica"].items()))
        blocos_criterio.append(
            f"CRITÉRIO: {c['nome']} (Peso: {c['peso']}%)\nRubrica:\n{rubrica}"
        )
    texto_criterios = "\n\n".join(blocos_criterio)

    if prompt_template is not None:
        contexto = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"CONTEXTO\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Candidato: {nome_candidato}\n\n"
            f"=== SCORECARD ===\n{texto_criterios}\n\n"
            f"=== TRANSCRIÇÃO LEGÍVEL (organizada em blocos de contexto) ===\n"
            f"Use para analisar o CONTEÚDO das respostas (raciocínio, exemplos, profundidade).\n"
            f"Cada bloco agrupa uma pergunta principal + a resposta completa do candidato + perguntas de aprofundamento relacionadas.\n"
            f"Os números de linha permitem citar trechos como evidência.\n{blocos}\n\n"
            f"=== TRANSCRIÇÃO ORIGINAL (fala real do candidato) ===\n"
            f"Use adicionalmente para critérios de COMUNICAÇÃO, clareza ou organização verbal.\n"
            f"Observe: hesitações, repetições, dificuldade de estruturar frases, respostas confusas.\n{original_numerada}"
        )
        prompt = contexto + "\n\n" + prompt_template + "\n\n" + _FORMAT_AVALIAR_CANDIDATO
    else:
        prompt = f"""Você é um entrevistador especialista em avaliação estruturada de candidatos. Avalie o candidato abaixo com base no scorecard fornecido.

Candidato: {nome_candidato}

=== SCORECARD ===
{texto_criterios}

=== TRANSCRIÇÃO LEGÍVEL (organizada em blocos de contexto) ===
Use para analisar o CONTEÚDO das respostas (raciocínio, exemplos, profundidade).
Cada bloco agrupa uma pergunta principal + a resposta completa do candidato + perguntas de aprofundamento relacionadas.
Os números de linha permitem citar trechos como evidência.
{blocos}

=== TRANSCRIÇÃO ORIGINAL (fala real do candidato) ===
Use adicionalmente para critérios de COMUNICAÇÃO, clareza ou organização verbal.
Observe: hesitações, repetições, dificuldade de estruturar frases, respostas confusas ou desorganizadas.
{original_numerada}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES — LEIA COM ATENÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Antes de atribuir qualquer nota: leia TODAS as evidências relevantes da transcrição e só então decida a nota.
Interprete a rubrica literalmente — use os descritores de cada nível exatamente como escritos.

Cada bloco agrupa uma pergunta principal + a resposta completa do candidato + perguntas de aprofundamento relacionadas.
Um novo bloco começa apenas quando o entrevistador muda de assunto com uma nova pergunta principal.

Para CADA critério do scorecard, siga este processo:

PASSO 1 — Leia os blocos completos e verifique se há evidência real na transcrição.
Procure falas do candidato que demonstrem diretamente o comportamento avaliado naquele critério.
Avalie sempre o raciocínio completo do candidato dentro do bloco — nunca com base em frases isoladas.

SE houver evidência suficiente → avalie com nota 1–5 conforme a rubrica (use o objeto "com nota").
SE NÃO houver evidência suficiente → NÃO atribua nota (use o objeto "sem evidência").

REGRA CRÍTICA: Ausência de evidência NÃO significa desempenho ruim.
Não invente contexto. Não assuma comportamentos não descritos na transcrição.
Só cite linhas que realmente existem na transcrição.

REGRAS PARA EVIDÊNCIAS:
- Cite o intervalo de linhas que contém a evidência. O sistema exibirá o trecho completo automaticamente.
  Formato de linha única:  "[12]"
  Formato de intervalo:    "[12-14]"
- SEMPRE prefira intervalos que abranjam a resposta completa — evite citar uma única linha quando a
  resposta se estende por várias linhas. Exemplo: use "[52-55]" em vez de "[53]".
- Nunca cite uma linha cujo significado dependa das linhas ao redor sem incluí-las no intervalo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATOS DOS OBJETOS DE AVALIAÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Objeto COM nota (quando há evidência para avaliar):
{{
  "criterio": "NomeDoCriterio",
  "nota": 3,
  "peso": 20,
  "contribuicao": 12.0,
  "evidencias": ["[12-14]"],
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
- evidencias: intervalo de linhas — linha única "[N]" ou intervalo "[N-M]" (sem texto adicional)
- lacunas: o que ficou ausente ou não foi demonstrado

Retorne APENAS JSON válido (sem markdown, sem explicação):
{{
  "avaliacoes": [
    {{ objeto com nota OU sem evidência para cada critério }}
  ]
}}"""

    texto = _chamar_api(prompt, max_tokens=6144, temperature=0)

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
