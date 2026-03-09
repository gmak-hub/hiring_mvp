# Cabine

Sistema de avaliação estruturada de candidatos com IA (Anthropic Claude).

---

## Visão Geral

O Cabine permite que equipes de RH:
- Criem cargos com descrições detalhadas
- Gerem scorecards de avaliação automaticamente com IA
- Submetam transcrições de entrevistas e recebam avaliações estruturadas
- Comparem candidatos lado a lado por critério

---

## Requisitos

- Python 3.11+
- Conta Anthropic (para geração de scorecards e avaliações com IA)
- Conta Supabase (para banco de dados em produção) — ou SQLite local para desenvolvimento

---

## Configuração Local (SQLite)

Passo mínimo para rodar localmente sem banco externo:

```bash
# 1. Clone o repositório
git clone <repo-url>
cd hiring_mvp

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure o .env
cp .env.example .env
# Edite .env e adicione sua ANTHROPIC_API_KEY
# Deixe DATABASE_URL comentado para usar SQLite local

# 5. Rode o servidor
uvicorn main:app --reload
```

Acesse: http://localhost:8000

---

## Configuração com Supabase (PostgreSQL)

No `.env`, defina:

```
DATABASE_URL=postgresql://postgres:[SUA-SENHA]@[SEU-HOST].supabase.co:5432/postgres
```

Como obter a URL:
1. Acesse [supabase.com](https://supabase.com) e abra seu projeto
2. Settings → Database → Connection string → URI
3. Substitua `[YOUR-PASSWORD]` pela senha do projeto

O sistema cria todas as tabelas automaticamente no primeiro boot.

---

## Variáveis de Ambiente

| Variável | Obrigatória | Descrição |
|---|---|---|
| `ANTHROPIC_API_KEY` | **Sim** | Chave da API Anthropic. Obtenha em console.anthropic.com |
| `DATABASE_URL` | Não | URL do banco. Padrão: `sqlite:///./hiring.db` |
| `SESSION_SECRET_KEY` | Não | Chave de assinatura das sessões. Troque em produção. |

---

## Como Rodar

```bash
# Desenvolvimento (com reload automático)
uvicorn main:app --reload

# Produção
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Credenciais Padrão

Na primeira execução, o sistema cria automaticamente um superadmin:

| Campo | Valor |
|---|---|
| Usuário | `admin` |
| Senha | `admin123` |
| Empresa | (deixe em branco no login) |

**Altere a senha imediatamente** após o primeiro login em Minha Conta.

---

## Perfis de Acesso

| Perfil | O que pode fazer |
|---|---|
| **Avaliador** | Acessar cargos e candidatos da empresa |
| **Admin da Empresa** | Tudo do Avaliador + criar/desativar usuários + excluir cargos/candidatos |
| **Super Admin** | Acesso total: ver todas as empresas, aprovar cadastros, impersonar usuários |

### Fluxo de cadastro de nova empresa

1. Qualquer pessoa acessa `/registro` e cria empresa + conta (ambos ficam **Pendentes**)
2. Super Admin acessa `/admin` e aprova a empresa — o admin da empresa é aprovado automaticamente
3. Admin da empresa faz login e pode criar usuários adicionais (já aprovados)

---

## Estrutura de Arquivos

```
hiring_mvp/
├── main.py           # Rotas FastAPI e lógica da aplicação
├── models.py         # Modelos SQLAlchemy (Company, User, Job, Candidate)
├── auth.py           # Autenticação, dependências FastAPI, hashing de senha
├── ai_client.py      # Integração Anthropic Claude (scorecard + avaliação)
├── requirements.txt  # Dependências Python
├── .env.example      # Template de variáveis de ambiente
├── static/
│   └── style.css
└── templates/
    ├── base.html
    ├── login.html
    ├── registro.html
    ├── conta.html
    ├── index.html
    ├── job_detail.html
    ├── candidate_detail.html
    ├── admin/
    │   └── dashboard.html
    └── empresa/
        └── usuarios.html
```
