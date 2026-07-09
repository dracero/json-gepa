# LangSmith QA Dashboard

Sistema full stack para extraer, curar, analizar y exportar datasets de preguntas y respuestas (QA) desde trazas de LangSmith. Diseñado específicamente para evaluar asistentes virtuales basados en RAG e interactuar mediante un flujo *Human-in-the-Loop* antes de alimentar pipelines de optimización (como DSPy GEPA).

---

## 📋 Tabla de Contenidos

- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Flujo de Datos y Extracción](#flujo-de-datos-y-extracción)
- [Extracción de Contexto RAG](#extracción-de-contexto-rag)
- [Pipeline de Análisis con IA (Multi-Agente)](#pipeline-de-análisis-con-ia-multi-agente)
- [Funcionalidades de Curación (Human-in-the-Loop)](#funcionalidades-de-curación-human-in-the-loop)
- [Instalación y Uso](#instalación-y-uso)
- [Configuración Avanzada](#configuración-avanzada)

---

## 🏗️ Arquitectura del Sistema

El sistema consta de tres componentes principales:

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
│  │  LangSmith   │  │  QA Analyst  │  │  API Router  │     │
│  │  Extractor   │─▶│ (LangGraph)  │─▶│  (REST API)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│   Extrae trazas      Pipeline con       Expone datos y      │
│   de LangGraph       4 agentes Gemini   analiza con IA      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Frontend (Astro.js)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   QA Cards   │  │  JSON Panel  │  │  AI Analysis │     │
│  │(Edición + TeX)│  │ (Raw Data)   │  │   (Gemini)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
             Optimización DSPy / GEPA
```

---

## 🔄 Flujo de Datos y Extracción

El modulo `LangSmithExtractor` conecta con la API de LangSmith para reconstruir las interacciones. Soporta dos estrategias de extracción complementarias:

### 1. Extracción Directa desde el Nodo Raíz (Recomendado para LangGraph)
Si la traza del proyecto (como en `RAG Histopatologia Pipeline`) guarda el estado final del turno en el nodo raíz, extraemos directamente desde sus `inputs` y `outputs`:
* **Pregunta del Alumno (`question`):** Extraída prioritariamente de `consulta_in` o `consulta_usuario`.
* **Respuesta del Asistente (`professor_response`):** Extraída de `respuesta_final`.
* **Contexto Recuperado (`context`):** Extraído directamente de `contexto_documentos` o `resultados_busqueda`.

Esto previene procesar de forma incorrecta los pasos de ejecución intermedia (como clasificadores de intención o optimizadores de queries) como si fueran respuestas dirigidas al estudiante.

### 2. Extracción por Sub-ejecuciones (Fallback para agentes secuenciales)
Si las trazas corresponden a llamadas tradicionales o pipelines lineales de LLMs:
1. Filtra los runs hijos del tipo `llm`.
2. Ordena cronológicamente los turnos por `start_time` ascendente.
3. Extrae el mensaje de usuario de los inputs y la generación del LLM de los outputs.

---

## 📂 Extracción de Contexto RAG

El extractor implementa un algoritmo en cascada para recuperar la información de contexto que el asistente RAG utilizó para responder:

1. **Salida del Estado Raíz (`contexto_documentos` / `resultados_busqueda`):** 
   - Utiliza la cadena de texto con formato de fuente y páginas (`contexto_documentos`).
   - Si no está formateado, lee `resultados_busqueda` (lista de diccionarios que contienen el `score`, `pdf_name`, y fragmentos de texto/imagen) y lo serializa automáticamente a texto estructurado.
   - Si existe `contexto_ontologico` asociado, lo concatena al final del bloque de contexto para dar una vista completa del RAG y la ontología aplicada.
2. **Runs del Retriever (Qdrant):** Si existen ejecuciones dedicadas de vector DB (tipo `retriever`), extrae los mejores matches de texto e imagen ponderando sus puntajes de similitud (`score`).
3. **Marcadores en Mensajes LLM:** Escanea los prompts en búsqueda de preámbulos estructurados del tipo `**SECCIONES DEL MANUAL:**`.
4. **Contexto Embebido en Historial:** Analiza bloques de preámbulo en el `HumanMessage` del LLM.

---

## 🧠 Pipeline de Análisis con IA (Multi-Agente)

El backend incorpora un servicio evaluador `QAAnalyst` estructurado como un **grafo multi-agente en LangGraph** impulsado por **Gemini 3.5 Flash**. Cuando se solicita analizar una interacción desde la interfaz, esta se procesa a través de 4 agentes especializados:

```
               ┌───────────────────────┐
               │    Entrada de Datos   │
               │ (Pregunta + RAG + Rta)│
               └───────────┬───────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Agente 1 · Pregunta   │
              │  (Evalúa claridad e    │
              │    intención del usuario)│
              └───────────┬───────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Agente 2 · Respuesta  │
              │   (Valora calidad y     │
              │  pedagogía de la rta)   │
              └───────────┬───────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    Agente 3 · Qdrant    │
              │  (Valora relevancia y   │
              │  completitud del RAG)   │
              └───────────┬───────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  Agente 4 · Sugerencias │
              │ (Propone mejoras y rta  │
              │ ideal para exportar)    │
              └─────────────────────────┘
```

1. **Agente de Pregunta (`question_analysis`):** Analiza la claridad, terminología y nivel de complejidad de la consulta del estudiante.
2. **Agente de Respuesta (`response_analysis`):** Evalúa la calidad pedagógica y didáctica del tutor de IA, asegurando que guíe al estudiante en vez de limitarse a resolver la consulta directamente.
3. **Agente de Recuperación (`retrieval_analysis`):** Revisa el contexto de documentos entregado por Qdrant, detectando si la información estaba incompleta, errónea o truncada en medio de una frase.
4. **Agente de Sugerencias (`improvement_suggestions`):** Proporciona mejoras detalladas y redacta una propuesta de respuesta ideal corregida.

---

## Curación en Interfaz (Human-in-the-Loop)

La interfaz en Astro.js cuenta con controles interactivos diseñados para pulir los datasets:

* **Visualización de Contexto Premium:** El contexto recuperado de Qdrant se muestra dentro de una caja con formato de código monoespaciado (`ui-monospace`), fondo diferenciado y un scroll vertical acotado (`max-height: 15rem`). Esto facilita examinar las fuentes, páginas y scores de similitud sin desbordar el diseño.
* **Soporte LaTeX Integrado:** Tanto las preguntas como las respuestas que contienen notación matemática o química (delimitadas por `$$...$$` y `$...$`) se renderizan en tiempo real mediante **KaTeX**.
* **Edición Directa:** El usuario puede hacer clic en **✏️ Editar** sobre cualquier respuesta para modificarla o redactarla a gusto. Al hacer clic en **💾 Guardar**, el dataset en memoria se actualiza con la respuesta ideal curada.
* **Exportador a DSPy:** Al pulsar **⬇ Exportar JSON**, se descarga un archivo `qa_dataset.json` con la estructura idónea de pares `question`, `context` y `professor_response` listos para ser entrenados y optimizados en DSPy.

---

## 🚀 Instalación y Uso

### Requisitos Previos
* **Python** ≥ 3.11 (con `uv` para gestión rápida de paquetes)
* **Node.js** ≥ 22

### Configuración del Entorno
Crea un archivo `.env` en la raíz del backend (`backend/.env`):
```bash
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=muevera_test
GOOGLE_API_KEY=AIzaSy...   # Requerido para la evaluación con Gemini
```

### Ejecutar en Desarrollo
Ejecuta el siguiente comando en la raíz del proyecto para iniciar frontend y backend de manera simultánea:
```bash
npm run dev
```

El backend se iniciará en `http://localhost:8000` y la interfaz del dashboard estará accesible en `http://localhost:4321`.
