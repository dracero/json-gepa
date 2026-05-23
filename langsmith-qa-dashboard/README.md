# LangSmith QA Dashboard

Sistema full stack para extraer, curar y exportar datasets de preguntas y respuestas desde trazas de LangSmith, diseñado para alimentar pipelines de optimización de prompts con DSPy GEPA.

---

## 📋 Tabla de Contenidos

- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Flujo de Datos](#flujo-de-datos)
- [Componentes Principales](#componentes-principales)
- [Interacción de Agentes](#interacción-de-agentes)
- [Human-in-the-Loop](#human-in-the-loop)
- [Integración con DSPy GEPA](#integración-con-dspy-gepa)
- [Instalación y Uso](#instalación-y-uso)
- [Configuración Avanzada](#configuración-avanzada)

---

## 🏗️ Arquitectura del Sistema

El sistema consta de tres capas principales:

```
┌─────────────────────────────────────────────────────────────┐
│                    LangSmith API                            │
│         (Trazas de interacciones estudiante-agente)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Backend (FastAPI)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  LangSmith   │  │   Context    │  │  API Router  │     │
│  │  Extractor   │─▶│  Inferrer    │─▶│  (REST API)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         │                  │                  │             │
│    Extrae runs       Infiere tema       Expone datos       │
│    de ChatGroq       y conceptos         como JSON         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Frontend (Astro.js)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   QA Cards   │  │  JSON Panel  │  │    Export    │     │
│  │  (KaTeX SSR) │  │   (Toggle)   │  │    Button    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│    Visualización      Inspección         Descarga          │
│    con LaTeX          del JSON           qa_dataset.json   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
                  DSPy GEPA Pipeline
```

---

## 🔄 Flujo de Datos

### 1. **Extracción desde LangSmith**

El `LangSmithExtractor` se conecta a la API de LangSmith y:

1. **Obtiene Root Runs** — Trazas raíz del proyecto configurado (default: `socratico_test`)
2. **Filtra Child Runs** — Busca runs hijos con `name="ChatGroq"` (el LLM que genera respuestas)
3. **Ordena cronológicamente** — Por `start_time` ascendente
4. **Extrae mensajes**:
   - `question`: último mensaje con `role="human"` o `type="user"` de `inputs.messages`
   - `agent_response`: contenido de `outputs.generations[0][0].text` o `outputs.content`
5. **Construye historial** — Lista acumulada de turnos anteriores en la misma traza

**Estructura de datos extraída:**

```python
Interaction(
    trace_id="019e2c00-...",
    trace_timestamp=datetime(...),
    turn_index=0,
    question="¿Cuál es la segunda ley de Newton?",
    agent_response="La segunda ley establece que F = ma...",
    history=[],  # vacío en el primer turno
)
```

### 2. **Inferencia de Contexto**

El `ContextInferrer` procesa cada `Interaction` y genera el campo `context` mediante **heurísticas de texto** (sin LLM externo):

**Algoritmo en dos pasos:**

#### Paso 1: Identificación del tema principal

- Mantiene un diccionario de 9 dominios con palabras clave:
  - Mecánica Clásica, Termodinámica, Electromagnetismo, Óptica, Mecánica Cuántica, Álgebra Lineal, Cálculo, Estadística y Probabilidad, Trabajo y Energía
- Cuenta coincidencias case-insensitive en `question + agent_response`
- Selecciona el tema con mayor puntaje (tie-break determinístico)
- Fallback: `"Física y Matemáticas"` si no hay coincidencias

#### Paso 2: Extracción de conceptos clave

Extrae hasta 5 conceptos únicos de:
1. **Expresiones LaTeX cortas** (`$...$` o `$$...$$`, < 30 chars)
2. **Palabras clave del tema** encontradas en el texto
3. **Notación especial** (e.g., `F = ma`, `E = mc²`)

**Formato de salida:**

```
"Mecánica Clásica. segunda ley de Newton, F = ma, fuerza, masa, aceleración"
```

### 3. **Generación del QA Dataset**

El endpoint `/qa-dataset` combina extracción + inferencia:

```python
QAItem(
    question="¿Cuál es la segunda ley de Newton?",
    context="Mecánica Clásica. segunda ley de Newton, F = ma, fuerza, masa",
    professor_response="La segunda ley establece que F = ma..."
)
```

**Propiedades garantizadas:**
- `question` y `professor_response` son copias exactas sin modificaciones
- `context` siempre tiene formato `"<Tema>. <Concepto1>, <Concepto2>, ..."`
- El tamaño del dataset = número de interacciones válidas

---

## 🧩 Componentes Principales

### Backend (Python + FastAPI)

#### `config.py` — Configuración con Pydantic

```python
class Settings(BaseSettings):
    langsmith_api_key: str  # Requerido
    langsmith_project: str = "socratico_test"
    model_config = {"env_file": ".env"}
```

- Lee variables de entorno desde `backend/.env`
- Lanza `ValidationError` si falta `LANGSMITH_API_KEY` → el servidor no inicia

#### `services/langsmith_extractor.py` — Extractor

**Métodos principales:**

- `get_interactions(project, start_time, end_time)` → `list[Interaction]`
- `_get_root_runs()` → llama a `client.list_runs(is_root=True)`
- `_get_child_runs()` → filtra por `filter='eq(name, "ChatGroq")'`
- `_extract_question()` → busca último mensaje `role="human"`
- `_extract_response()` → extrae de `outputs.generations` o `outputs.content`

**Manejo de errores:**
- Runs sin mensaje humano → skip con warning
- Runs sin respuesta → skip con warning
- Errores de conexión → propaga como `LangSmithError`

#### `services/context_inferrer.py` — Inferrer

**Diccionario de dominios:**

```python
TOPIC_KEYWORDS = {
    "Mecánica Clásica": ["fuerza", "masa", "aceleración", ...],
    "Termodinámica": ["temperatura", "calor", "entropía", ...],
    # ... 7 dominios más
}
```

**Métodos:**
- `infer(question, agent_response)` → `str` (formato `"<Tema>. <Conceptos>"`)
- `_identify_topic(text)` → conteo de keywords
- `_extract_concepts(text, topic)` → LaTeX + keywords + notación especial

#### `routers/api.py` — Endpoints REST

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/interactions` | Lista de `Interaction` cruda |
| `GET` | `/qa-dataset` | Lista de `QAItem` con contexto inferido |
| `POST` | `/qa-dataset/export` | Descarga `qa_dataset.json` |
| `GET` | `/debug/runs` | Inspección de estructura cruda (debug) |

**Parámetros de query comunes:**
- `project`: nombre del proyecto LangSmith (override del default)
- `start_time`: ISO 8601 datetime (default: últimas 48h)
- `end_time`: ISO 8601 datetime (default: now)

### Frontend (Astro.js + TypeScript)

#### `src/lib/api.ts` — Cliente HTTP

```typescript
export async function fetchQADataset(params?: QueryParams): Promise<QAItem[]>
export async function fetchInteractions(params?: QueryParams): Promise<Interaction[]>
export async function exportDataset(items: QAItem[]): Promise<Blob>
```

- Base URL configurable via `PUBLIC_API_URL` (default: `http://localhost:8000`)
- Manejo de errores HTTP con mensajes descriptivos

#### `src/components/QACard.astro` — Tarjeta de Q&A

- **Renderizado LaTeX SSR** con KaTeX:
  - `$$...$$` → display mode (bloque)
  - `$...$` → inline mode
  - `throwOnError: false` → muestra texto original si la fórmula es inválida
- **Estructura visual:**
  - Header: número de tarjeta + badge de contexto
  - Pregunta: texto con LaTeX renderizado
  - Respuesta: texto con LaTeX renderizado + estilos para párrafos

#### `src/components/ViewToggle.astro` — Toggle de vista

- Botón que alterna entre vista de tarjetas y vista JSON
- Script de isla (client-side) que controla `display` de `[data-testid="card-view"]` y `[data-testid="json-view"]`

#### `src/components/JsonPanel.astro` — Panel JSON

- Muestra el dataset como JSON formateado (`JSON.stringify(..., null, 2)`)
- Oculto por defecto (`display:none`)
- Resaltado de sintaxis con estilos CSS

#### `src/components/ExportButton.astro` — Botón de exportación

- Lee el dataset desde `window.__QA_DATASET__` (inyectado por `index.astro`)
- Llama a `exportDataset()` → `POST /qa-dataset/export`
- Crea un `Blob` y dispara descarga con `URL.createObjectURL`
- Muestra mensaje informativo si el dataset está vacío

#### `src/pages/index.astro` — Página principal

- **SSR**: hace fetch a `/qa-dataset` en el servidor
- **Manejo de estados:**
  - Error de conexión → banner rojo con mensaje descriptivo
  - Dataset vacío → estado vacío con icono y hint
  - Dataset con datos → renderiza tarjetas + controles
- **Inyección de datos:** `<script define:vars={{ qaDataset: items }}>` expone el dataset al cliente

---

## 🤖 Interacción de Agentes

### Agente Estudiante (Usuario Final)

Interactúa con un agente conversacional (e.g., ChatGroq) que responde preguntas de física/matemáticas. Cada interacción genera una traza en LangSmith con:

- **Input**: mensaje del estudiante
- **Output**: respuesta del agente (con LaTeX, explicaciones, etc.)
- **Metadata**: timestamp, trace_id, run_id

### Agente Docente (Sistema LangSmith QA Dashboard)

**Rol**: Extrae, procesa y presenta las interacciones para revisión humana.

**Flujo de trabajo:**

1. **Extracción automática** (cada vez que se carga el dashboard):
   - Conecta a LangSmith API
   - Filtra runs de ChatGroq en las últimas 48h
   - Extrae preguntas y respuestas

2. **Procesamiento automático**:
   - Infiere tema y conceptos con heurísticas
   - Genera el campo `context` para cada QA pair

3. **Presentación al humano**:
   - Renderiza tarjetas visuales con LaTeX
   - Permite inspección del JSON crudo
   - Ofrece exportación con un clic

### Agente de Optimización (DSPy GEPA)

**Rol**: Consume el dataset exportado para optimizar prompts.

**Input esperado**: archivo `qa_dataset.json` con formato:

```json
[
  {
    "question": "...",
    "context": "...",
    "professor_response": "..."
  }
]
```

**Uso en DSPy GEPA:**

```python
import json
from dspy import Example

# Cargar dataset
with open("qa_dataset.json") as f:
    qa_data = json.load(f)

# Convertir a ejemplos de DSPy
trainset = [
    Example(
        question=item["question"],
        context=item["context"]
    ).with_inputs("question", "context")
    for item in qa_data
]

# Usar en optimización
optimizer = GEPA(metric=answer_quality_metric)
optimized_program = optimizer.compile(
    student=PhysicsTeacher(),
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=8,
)
```

---

## 👤 Human-in-the-Loop

El sistema está diseñado para **revisión y curación manual** antes de alimentar DSPy GEPA:

### Flujo de Curación

```
┌─────────────────────────────────────────────────────────────┐
│  1. Extracción Automática                                   │
│     LangSmith → Backend → Dataset crudo                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Revisión Humana (Dashboard)                             │
│     ┌─────────────────────────────────────────────┐         │
│     │  Vista de Tarjetas                          │         │
│     │  • Revisar preguntas y respuestas           │         │
│     │  • Verificar renderizado LaTeX              │         │
│     │  • Identificar errores o respuestas pobres  │         │
│     └─────────────────────────────────────────────┘         │
│     ┌─────────────────────────────────────────────┐         │
│     │  Vista JSON                                 │         │
│     │  • Inspeccionar estructura                  │         │
│     │  • Verificar campos context                 │         │
│     │  • Detectar datos faltantes                 │         │
│     └─────────────────────────────────────────────┘         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Decisión Humana                                         │
│     ¿El dataset es de calidad suficiente?                   │
│     ├─ SÍ → Exportar qa_dataset.json                        │
│     └─ NO → Ajustar agente/prompts, regenerar trazas       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Curación Manual (Opcional)                              │
│     • Editar qa_dataset.json en editor de texto             │
│     • Corregir contextos mal inferidos                      │
│     • Eliminar ejemplos de baja calidad                     │
│     • Agregar ejemplos sintéticos                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Alimentar DSPy GEPA                                     │
│     optimizer.compile(trainset=curated_examples)            │
└─────────────────────────────────────────────────────────────┘
```

### Puntos de Intervención Humana

#### A. **Revisión Visual (Dashboard)**

**Objetivo**: Detectar problemas antes de exportar.

**Checklist de revisión:**
- [ ] ¿Las preguntas son claras y completas?
- [ ] ¿Las respuestas son correctas y bien explicadas?
- [ ] ¿El LaTeX se renderiza correctamente?
- [ ] ¿El campo `context` es relevante?
- [ ] ¿Hay respuestas truncadas o errores de extracción?

**Acciones:**
- Si hay problemas → ajustar el agente conversacional o los prompts
- Si todo está bien → exportar

#### B. **Edición Manual del JSON**

**Objetivo**: Refinar el dataset antes de DSPy GEPA.

**Casos de uso:**

1. **Corregir contextos mal inferidos:**

```json
// Antes (inferido automáticamente)
{
  "context": "Física y Matemáticas. trabajo, energía"
}

// Después (corregido manualmente)
{
  "context": "Trabajo y Energía. fuerza conservativa, trayectoria cerrada, integral de línea"
}
```

2. **Eliminar ejemplos de baja calidad:**

```json
// Eliminar respuestas vagas o incorrectas
[
  {"question": "...", "context": "...", "professor_response": "No estoy seguro..."},  // ❌ Eliminar
  {"question": "...", "context": "...", "professor_response": "Excelente pregunta! ..."}  // ✅ Mantener
]
```

3. **Agregar ejemplos sintéticos:**

```json
// Agregar manualmente ejemplos de alta calidad
{
  "question": "¿Qué es el momento angular?",
  "context": "Mecánica Clásica. momento angular, L = r × p, conservación",
  "professor_response": "El momento angular L es el producto vectorial..."
}
```

#### C. **Validación Post-Exportación**

**Script de validación** (ejemplo):

```python
import json

def validate_qa_dataset(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    issues = []
    for i, item in enumerate(data):
        # Verificar campos requeridos
        if not item.get("question"):
            issues.append(f"Item {i}: missing question")
        if not item.get("context"):
            issues.append(f"Item {i}: missing context")
        if not item.get("professor_response"):
            issues.append(f"Item {i}: missing professor_response")
        
        # Verificar formato de context
        if ". " not in item.get("context", ""):
            issues.append(f"Item {i}: context format invalid")
        
        # Verificar longitud mínima de respuesta
        if len(item.get("professor_response", "")) < 50:
            issues.append(f"Item {i}: response too short")
    
    return issues

# Uso
issues = validate_qa_dataset("qa_dataset.json")
if issues:
    print("⚠️  Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ Dataset is valid!")
```

---

## 🔗 Integración con DSPy GEPA

### Formato de Entrada

DSPy GEPA espera ejemplos con:
- `question`: la pregunta del estudiante
- `context`: información de dominio (tema + conceptos)
- `answer` (opcional): la respuesta esperada (para métricas)

El campo `professor_response` del dataset puede usarse como:
1. **Ground truth** para métricas de evaluación
2. **Ejemplo de referencia** para bootstrapping
3. **Contexto adicional** para el optimizador

### Ejemplo de Integración

```python
import json
import dspy
from dspy.teleprompt import GEPA

# 1. Cargar dataset exportado
with open("qa_dataset.json") as f:
    qa_data = json.load(f)

# 2. Convertir a ejemplos de DSPy
trainset = []
for item in qa_data:
    example = dspy.Example(
        question=item["question"],
        context=item["context"],
        reference_answer=item["professor_response"]  # para métricas
    ).with_inputs("question", "context")
    trainset.append(example)

# 3. Definir el programa a optimizar
class PhysicsTeacher(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question, context -> answer")
    
    def forward(self, question, context):
        return self.generate_answer(question=question, context=context)

# 4. Definir métrica de calidad
def answer_quality(example, pred, trace=None):
    # Comparar con reference_answer usando LLM judge
    judge_prompt = f"""
    Question: {example.question}
    Reference: {example.reference_answer}
    Prediction: {pred.answer}
    
    Rate the prediction quality (0-10):
    """
    score = dspy.Predict("prompt -> score")(prompt=judge_prompt).score
    return float(score) / 10.0

# 5. Optimizar con GEPA
optimizer = GEPA(
    metric=answer_quality,
    breadth=10,
    depth=3,
    init_temperature=1.4,
)

optimized_teacher = optimizer.compile(
    student=PhysicsTeacher(),
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=8,
)

# 6. Evaluar
devset = trainset[:20]  # subset para evaluación
dspy.Evaluate(
    devset=devset,
    metric=answer_quality,
    display_progress=True,
)(optimized_teacher)
```

### Flujo Iterativo

```
┌─────────────────────────────────────────────────────────────┐
│  Iteración 1                                                │
│  1. Exportar dataset inicial (50 ejemplos)                 │
│  2. Optimizar con GEPA                                      │
│  3. Evaluar performance → 60% accuracy                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Iteración 2                                                │
│  1. Identificar casos fallidos                              │
│  2. Generar más trazas en esos dominios                     │
│  3. Exportar dataset ampliado (100 ejemplos)                │
│  4. Re-optimizar con GEPA                                   │
│  5. Evaluar performance → 75% accuracy                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Iteración 3                                                │
│  1. Curar manualmente ejemplos difíciles                    │
│  2. Agregar ejemplos sintéticos de alta calidad             │
│  3. Exportar dataset final (120 ejemplos)                   │
│  4. Optimización final con GEPA                             │
│  5. Evaluar performance → 85% accuracy ✅                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Instalación y Uso

### Requisitos Previos

- **Python** ≥ 3.11 (con `uv` instalado)
- **Node.js** ≥ 22.12.0
- **LangSmith API Key** (obtener en [smith.langchain.com/settings](https://smith.langchain.com/settings))

### Instalación

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd langsmith-qa-dashboard

# 2. Instalar dependencias (backend + frontend)
npm install

# 3. Configurar variables de entorno
cp backend/.env.example backend/.env
# Editar backend/.env y poner tu LANGSMITH_API_KEY real
```

### Ejecución

```bash
# Desde la raíz del proyecto
npm run dev
```

Esto levanta:
- **Backend**: `http://localhost:8000` (FastAPI + uvicorn)
- **Frontend**: `http://localhost:4321` (Astro.js dev server)

### Uso del Dashboard

1. **Abrir** `http://localhost:4321` en el navegador
2. **Revisar** las tarjetas de Q&A (con LaTeX renderizado)
3. **Alternar** a vista JSON para inspeccionar la estructura
4. **Exportar** el dataset con el botón "⬇ Exportar JSON"
5. **Guardar** el archivo `qa_dataset.json` descargado

### Debug de Extracción

Si el dashboard muestra "No se encontraron interacciones", inspeccionar la estructura cruda:

```bash
# Abrir en el navegador
http://localhost:8000/debug/runs
```

Esto muestra:
- Keys de `inputs` y `outputs` de los runs
- Roles de los mensajes (`human`, `ai`, etc.)
- Preview del contenido

Usar esta info para ajustar `_extract_question` y `_extract_response` si el formato de LangSmith cambió.

---

## ⚙️ Configuración Avanzada

### Variables de Entorno

**Backend** (`backend/.env`):

```bash
# Requerido
LANGSMITH_API_KEY=lsv2_pt_...

# Opcional
LANGSMITH_PROJECT=socratico_test  # proyecto por defecto
```

**Frontend** (`.env` en raíz o `frontend/.env`):

```bash
# Opcional
PUBLIC_API_URL=http://localhost:8000  # URL del backend
```

### Ajustar Rango de Tiempo

Por defecto, el dashboard extrae las últimas 48 horas. Para cambiar:

**Opción A — Query params en el frontend:**

Editar `frontend/src/pages/index.astro`:

```typescript
const res = await fetch(`${API_URL}/qa-dataset?start_time=2026-05-01T00:00:00Z&end_time=2026-05-16T23:59:59Z`);
```

**Opción B — Default en el backend:**

Editar `backend/services/langsmith_extractor.py`:

```python
def _compute_default_time_range(...):
    now = datetime.utcnow()
    start = start_time if start_time is not None else now - timedelta(days=7)  # cambiar a 7 días
    end = end_time if end_time is not None else now
    return start, end
```

### Personalizar Dominios del Context_Inferrer

Editar `backend/services/context_inferrer.py`:

```python
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Mecánica Clásica": ["fuerza", "masa", ...],
    "Tu Nuevo Dominio": ["keyword1", "keyword2", ...],  # agregar aquí
    # ...
}
```

### Cambiar el Filtro de Runs

Por defecto filtra por `name="ChatGroq"`. Para cambiar:

Editar `backend/services/langsmith_extractor.py`:

```python
def _get_child_runs(self, root_run_id: str, project: str):
    runs = list(
        self.client.list_runs(
            project_name=project,
            trace_id=root_run_id,
            filter='eq(name, "TuModeloLLM")',  # cambiar aquí
        )
    )
    return runs
```

### Despliegue en Producción

**Backend:**

```bash
cd backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend:**

```bash
cd frontend
npm run build
node dist/server/entry.mjs
```

O usar un adaptador de Astro para tu plataforma (Vercel, Netlify, Docker, etc.).

---

## 📚 Referencias

- **LangSmith API**: [docs.smith.langchain.com](https://docs.smith.langchain.com)
- **DSPy**: [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- **GEPA Optimizer**: [dspy.ai/teleprompt/gepa](https://dspy.ai/teleprompt/gepa)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **Astro.js**: [astro.build](https://astro.build)
- **KaTeX**: [katex.org](https://katex.org)

---

## 🤝 Contribuciones

Para reportar bugs o sugerir mejoras, abrir un issue en el repositorio.

---

## 📄 Licencia

[Especificar licencia aquí]
