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

from anthropic import Anthropic, APIConnectionError, APIStatusError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

MODELO = "claude-sonnet-4-6"

MAPA_CONFIANCA = {
    "baixo": "Baixo", "low": "Baixo",
    "médio": "Médio", "medio": "Médio", "medium": "Médio",
    "alto": "Alto", "high": "Alto",
}

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

    # Normalize weights to sum to exactly 100
    total = sum(c["peso"] for c in criterios)
    if total != 100:
        for c in criterios:
            c["peso"] = round(c["peso"] * 100 / total)
        diferenca = 100 - sum(c["peso"] for c in criterios)
        criterios[0]["peso"] += diferenca

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

Para CADA um dos 5 critérios acima, produza uma avaliação completa.

REGRAS:
- Atribua nota 1–5 com base ESTRITAMENTE na definição da rubrica.
- contribuicao = (nota × peso) / 5  [máximo por critério = peso; máximo total = 100]
- confianca: "Baixo" se nenhuma ou mínima evidência; "Médio" se evidência parcial; "Alto" se evidência clara e consistente.
- evidencias: lista de CITAÇÕES DIRETAS da transcrição com número de linha, no formato: "[LINHA] 'citação exata'"
- lacunas: declare explicitamente o que NÃO foi demonstrado ou qual evidência estava ausente ou insuficiente.
- Se NÃO houver evidência relevante para um critério: nota=2, confianca="Baixo", evidencias=[], lacunas="[descreva o que era esperado, mas não foi encontrado na transcrição]"
- NUNCA invente evidências. Cite apenas linhas que existem na transcrição.
- nota_final = soma de todos os valores de contribuicao (faixa teórica: 20–100)

Retorne APENAS JSON válido (sem markdown, sem explicação):
{{
  "avaliacoes": [
    {{
      "criterio": "NomeDoCriterio",
      "nota": 3,
      "peso": 20,
      "contribuicao": 12.0,
      "confianca": "Médio",
      "evidencias": ["[12] 'citação exata da linha 12'"],
      "lacunas": "Não demonstrou X nem Y"
    }}
  ],
  "nota_final": 64.0
}}"""

    texto = _chamar_api(prompt, max_tokens=6144)

    try:
        bruto = _extrair_json(texto)
        dados = json.loads(bruto)
    except json.JSONDecodeError as e:
        print(f"[AI PARSING ERROR] Resposta bruta da avaliação:\n{texto}")
        raise AIParsingError(f"A IA retornou JSON inválido na avaliação: {e}") from e

    # Normalize confidence labels and recalculate weighted contributions server-side
    for av in dados["avaliacoes"]:
        val = av.get("confianca", "").lower().strip()
        av["confianca"] = MAPA_CONFIANCA.get(val, "Baixo")
        av["contribuicao"] = round((av["nota"] * av["peso"]) / 5, 1)

    nota_final = sum(av["contribuicao"] for av in dados["avaliacoes"])
    dados["nota_final"] = round(nota_final, 1)

    return dados
