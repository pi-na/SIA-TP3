---
name: specs and plans location
description: Where superpowers specs and plans live in this repo (they are gitignored)
type: reference
originSessionId: 6a5ca89e-5f83-4840-90c3-26115775244c
---
Superpowers specs y plans para este proyecto están en `docs/superpowers/` en la raíz del repo (`/home/nico/Desktop/SIA/SIA-TP3/docs/superpowers/`):

- `docs/superpowers/specs/` — specs aprobados (output de brainstorming)
- `docs/superpowers/plans/` — plans de implementación (output de writing-plans)

**Importante:** la carpeta `docs/superpowers/` está en `.gitignore` (decisión heredada del commit `f6b736c` de Tomás). Por lo tanto:
- Los archivos existen solo en la copia local del usuario.
- **No aparecen en worktrees creados con `git worktree add`** — solo viven en el directorio raíz del repo.
- Cuando trabajes desde un worktree, leé el plan vía path absoluto (`/home/nico/Desktop/SIA/SIA-TP3/docs/superpowers/...`).

Naming convention observada: `YYYY-MM-DD-<slug>.md` para specs y plans.
