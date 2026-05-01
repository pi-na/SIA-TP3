# Handoff bundle — Resumir TP3 completion en otra computadora

Este directorio contiene todo lo que necesitás para retomar el trabajo del branch `tp3-mlp` en otra PC. **No es código del proyecto** — es un paquete temporal de contexto. Una vez retomado, podés borrar este directorio.

## Qué hay acá

| Archivo | Para qué sirve |
|---|---|
| `spec.md` | Diseño aprobado del trabajo (corresponde a `docs/superpowers/specs/2026-05-01-tp3-completion-design.md`) |
| `plan.md` | Plan de implementación (39 tareas, corresponde a `docs/superpowers/plans/2026-05-01-tp3-completion.md`) |
| `MEMORY.md` + `*_status.md` + `specs_and_plans_location.md` | Memoria de Claude para esta sesión (van en `~/.claude/projects/-home-nico-Desktop-SIA-SIA-TP3/memory/`) |

## Cómo usarlo en la nueva PC

### 1. Clonar y poner los archivos en su lugar

```bash
git clone git@github.com:pi-na/SIA-TP3.git
cd SIA-TP3
git checkout tp3-mlp
git pull

# Restaurar spec/plan a su ubicación gitignored (para que Claude las lea naturalmente):
mkdir -p docs/superpowers/specs docs/superpowers/plans
cp _handoff_resume/spec.md docs/superpowers/specs/2026-05-01-tp3-completion-design.md
cp _handoff_resume/plan.md docs/superpowers/plans/2026-05-01-tp3-completion.md

# Restaurar memoria de Claude:
mkdir -p ~/.claude/projects/-home-nico-Desktop-SIA-SIA-TP3/memory
cp _handoff_resume/MEMORY.md ~/.claude/projects/-home-nico-Desktop-SIA-SIA-TP3/memory/
cp _handoff_resume/specs_and_plans_location.md ~/.claude/projects/-home-nico-Desktop-SIA-SIA-TP3/memory/
cp _handoff_resume/tp3_completion_status.md ~/.claude/projects/-home-nico-Desktop-SIA-SIA-TP3/memory/
```

**OJO con el path**: el directorio de memoria depende del path absoluto del proyecto. Si en la nueva PC el repo NO está en `/home/nico/Desktop/SIA/SIA-TP3`, ajustá el path. Para ver el path correcto, abrí Claude Code en el repo y ejecutá `pwd` — la memoria va en `~/.claude/projects/<path-con-guiones>/memory/`.

### 2. Setup del entorno

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib pytest
.venv/bin/python -m pytest mlp/tests/ ejercicio0/tests/  # debería dar 78 passed
```

### 3. (Opcional) Crear el worktree de nuevo

El branch `tp3-mlp` ya está en remote. Podés trabajar directo sobre él, o crear un worktree:

```bash
git worktree add .worktrees/tp3-mlp tp3-mlp
cd .worktrees/tp3-mlp
```

### 4. Borrar este directorio (una vez restaurado todo)

```bash
git rm -r _handoff_resume/
git commit -m "chore: remove handoff bundle after resume"
```

### 5. Decirle a Claude qué hacer

Abrí Claude Code en el repo y decile algo como:

> "Estoy retomando el plan de TP3 completion en una PC nueva. La memoria está restaurada en `~/.claude/projects/.../memory/`, el spec en `docs/superpowers/specs/`, el plan en `docs/superpowers/plans/`. La próxima tarea es Task 27 — correr los sweeps de Fase 1 de Ej2 y elegir `base.json`. Continuá."

Claude debería leer la memoria, encontrar el plan, y arrancar con Task 27.

## Estado al momento del handoff

- **25 tareas + Task 26 completas.** 78/78 tests pasan en `mlp/tests/` y `ejercicio0/tests/`.
- **Próxima tarea: Task 27** — correr 16 sweeps de Fase 1 (arch + opt + lr + batch), elegir `base.json`, verificar con K-fold=5. Estimado ~60-90 min de cómputo.
- arch_50 ya corrió en la PC original (val_acc 95.78%) pero el output NO está en el repo (gitignored). En la nueva PC, simplemente correr todo de cero.
