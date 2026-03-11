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
FORMATOS DOS OBJETOS DE AVALIAÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Objeto COM nota (quando há evidência para avaliar):
{
  "criterio": "NomeDoCriterio",
  "nota": 3,
  "peso": 20,
  "contribuicao": 12.0,
  "evidencias": ["[linhas 12–18] 'trecho exato das linhas 12 a 18'"],
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
- evidencias: TRECHOS CONTÍNUOS referenciados como "[linhas X–Y] 'trecho exato'"
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


def _ordenar_evidencias(evidencias: list[str]) -> list[str]:
    """Sort evidence citations by start line number.

    Accepts two formats:
    - "[linhas X–Y] 'trecho'"  (new range format, em-dash or hyphen)
    - "[N] 'citação'"          (legacy single-line format)
    """
    def _linha(ev: str) -> int:
        # New format: [linhas X–Y] or [linhas X-Y]
        m = re.match(r"\[linhas\s+(\d+)[–\-]", ev.strip())
        if m:
            return int(m.group(1))
        # Legacy format: [N]
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

Ao citar evidências, use sempre intervalos contínuos de linhas no formato "[linhas X–Y] 'trecho exato'".
Interrupções curtas do entrevistador não quebram o trecho — mantenha o intervalo contínuo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO A — com evidência (use quando há base na transcrição para avaliar):
{{
  "criterio": "{nome_c}",
  "nota": 3,
  "peso": {peso_c},
  "contribuicao": {round(3 * peso_c / 5, 1)},
  "evidencias": ["[linhas 12–18] 'trecho exato das linhas 12 a 18'"],
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


def _gerar_resumo(nome_candidato: str, avaliacoes: list) -> str:
    """Gera um resumo de 3–4 frases do candidato a partir da avaliação já feita (não da transcrição)."""
    linhas = []
    for av in avaliacoes:
        if av.get("sem_evidencia"):
            linhas.append(
                f"- {av['criterio']} (peso {av['peso']}%): evidência insuficiente — {av.get('motivo', '')}"
            )
        else:
            lacunas = f" Lacunas: {av['lacunas']}" if av.get("lacunas") else ""
            linhas.append(
                f"- {av['criterio']} (peso {av['peso']}%, nota {av['nota']}/5).{lacunas}"
            )
    avaliacao_texto = "\n".join(linhas)

    prompt = f"""Com base na avaliação estruturada do candidato {nome_candidato}, escreva um resumo de 3 a 4 frases.

AVALIAÇÃO:
{avaliacao_texto}

REGRAS:
- Máximo de 4 frases curtas e diretas
- Objetivo e informativo — sem linguagem corporativa vaga ("excelente profissional", "grande potencial", etc.)
- Sintetize o desempenho geral e os principais comportamentos observados na entrevista
- Se houver limitação ou risco relevante, mencione-o
- Baseie-se apenas nos dados acima — não invente comportamentos
- Retorne apenas o texto corrido, sem título nem formatação"""

    return _chamar_api(prompt, max_tokens=300).strip()


def avaliar_candidato(scorecard: dict, nome_candidato: str, transcricao: str, prompt_template: str | None = None) -> dict:
    numerada = _numerar_linhas(transcricao)

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
            f"=== TRANSCRIÇÃO (com número de linhas) ===\n{numerada}"
        )
        prompt = contexto + "\n\n" + prompt_template + "\n\n" + _FORMAT_AVALIAR_CANDIDATO
    else:
        prompt = f"""Você é um entrevistador especialista em avaliação estruturada de candidatos. Avalie o candidato abaixo com base no scorecard fornecido.

Candidato: {nome_candidato}

=== SCORECARD ===
{texto_criterios}

=== TRANSCRIÇÃO (com número de linhas) ===
{numerada}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES — SIGA AS 4 ETAPAS ABAIXO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ETAPA 1 — IDENTIFICAR TRECHOS DE RACIOCÍNIO
Leia a transcrição inteira e identifique trechos contínuos onde o candidato está explicando a mesma ideia ou experiência.
Regras:
- Interrupções curtas do entrevistador (perguntas de clarificação, "uhm", "entendi") NÃO quebram o trecho.
- Mudanças claras de assunto iniciam um novo trecho.
- Cada trecho deve ser referenciado como [linhas X–Y].

ETAPA 2 — MAPEAR CRITÉRIOS POR TRECHO
Para cada trecho identificado, verifique quais critérios do scorecard aparecem nele.
Regras:
- Cada trecho pode impactar NO MÁXIMO 2 critérios.
- Se mais de dois critérios parecerem relevantes, escolha apenas os DOIS mais fortes.
- Marque apenas critérios claramente demonstrados — não force associações vagas.

ETAPA 3 — AGRUPAR EVIDÊNCIAS POR CRITÉRIO
Reúna todos os trechos mapeados para cada critério do scorecard.
- Se um critério não tiver nenhum trecho mapeado, ele será avaliado como "sem evidência".
- Cada evidência deve ser referenciada como [linhas X–Y] 'trecho exato'.

ETAPA 4 — AVALIAR CADA CRITÉRIO
Para cada critério, com base nas evidências agrupadas na etapa anterior:

Antes de usar qualquer trecho como evidência, classifique mentalmente o tipo de conteúdo:
  (1) Evidência comportamental direta — o candidato descreve algo que fez, decidiu ou executou.
  (2) Relato de feedback ou autocrítica — o candidato conta algo que disseram sobre ele, ou uma crítica que recebeu.
  (3) Reflexão ou opinião — o candidato fala de forma hipotética, abstrata ou genérica.

Regras de classificação:
- Apenas trechos do tipo (1) devem ser usados como evidência forte.
- Trechos do tipo (2) NÃO são evidência automática de fraqueza. Quando o candidato menciona um feedback negativo, avalie: ele demonstra aprendizado? Explica como mudou? Se sim, o foco deve ser reflexão e evolução — não a crítica em si. Nunca trate uma frase negativa isolada como prova de incompetência quando está dentro de uma pergunta de feedback, autocrítica ou reflexão sobre erros.
- Só considere fraqueza real quando houver descrição clara de comportamento recorrente ou exemplo concreto que demonstre a falha em ação.

- Compare as evidências com a rubrica do critério.
- SE houver evidência suficiente → atribua nota 1–5 e retorne o objeto "com nota".
- SE NÃO houver evidência suficiente → NÃO atribua nota e retorne o objeto "sem evidência".

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
  "evidencias": ["[linhas 12–18] 'trecho exato das linhas 12 a 18'"],
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
- evidencias: TRECHOS CONTÍNUOS referenciados como "[linhas X–Y] 'trecho exato'"
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
    dados["resumo"] = _gerar_resumo(nome_candidato, dados["avaliacoes"])

    return dados
