# HiringEval MVP — Memory

## Stack
- FastAPI + SQLite (SQLAlchemy ORM) + Jinja2 templates + Anthropic Claude
- No frontend framework — pure HTML/CSS with inline JS
- `passlib[bcrypt]` for password hashing, `SessionMiddleware` for auth sessions

## Key Files
- `main.py` — all routes and app setup
- `models.py` — SQLAlchemy models + DB migration logic
- `auth.py` — password utils, FastAPI auth dependencies
- `ai_client.py` — Anthropic API calls (scorecard + evaluation)
- `static/style.css` — full design system
- `templates/` — Jinja2 templates (base.html, login.html, index.html, job_detail.html, candidate_detail.html, admin/dashboard.html, empresa/usuarios.html)

## Data Models
- `Company` — empresa (id, name, is_active, created_at)
- `User` — (id, company_id nullable, username unique, name, password_hash, role, is_active)
- `Job` — (id, company_id, name, description, scorecard JSON, is_deleted, deleted_at, created_at)
- `Candidate` — (id, job_id, name, transcript, evaluation JSON, final_score, is_deleted, deleted_at)

## Auth & Permissions
- Session via `SessionMiddleware` (cookie-based, signed)
- `session["user_id"]` = actual logged-in user
- `session["impersonating_user_id"]` = if superadmin is impersonating
- Roles: `avaliador` | `admin` | `superadmin`
- `get_current_user` → effective user (impersonated if set)
- `get_actual_user` → real logged-in user (ignores impersonation)
- `require_admin` → role in (admin, superadmin)
- `require_superadmin` → actual user must be superadmin
- `RequiresLoginException` → triggers redirect to /login via exception handler

## Default Credentials (first boot)
- username: `admin`, password: `admin123`, role: `superadmin`

## Multi-tenancy
- All data queries filter `company_id == current_user.company_id` for non-superadmin
- Superadmin sees all data (no company filter)
- When impersonating, data is scoped to the impersonated user's company

## Soft Delete
- `is_deleted` (Boolean, default False) + `deleted_at` (DateTime) on Job and Candidate
- Deleted items excluded from all normal queries
- Only admin/superadmin can delete

## Migration
- `_migrate_db()` in models.py adds new columns to existing tables without destroying data
- Runs on every startup (safe — checks for column existence first)
- First boot seeds: default company, assigns orphan jobs to it, creates superadmin

## Routes Overview (20 total)
- Auth: GET/POST /login, GET /logout
- Jobs: GET /, GET/POST /cargos/novo, GET /cargos/{id}, POST /cargos/{id}/gerar-scorecard, POST /cargos/{id}/excluir
- Candidates: GET/POST /cargos/{id}/candidatos/novo, GET /cargos/{id}/candidatos/{id}, POST /cargos/{id}/candidatos/{id}/excluir
- Super Admin: GET /admin, POST /admin/empresas, POST /admin/impersonar/{id}, POST /admin/parar-impersonar
- Company Admin: GET/POST /empresa/usuarios, POST /empresa/usuarios/{id}/desativar

## Template Context
- All authenticated routes use `ctx(request, current_user, **extra)` helper
- Passes: `request`, `user` (current_user), `is_impersonating` (bool)
- Templates use `user.role` to conditionally show admin buttons
