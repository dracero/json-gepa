# LangSmith QA Dashboard & Pedagogical Agent Evaluator

## 📌 Descripción del Proyecto

Este proyecto consiste en un entorno de visualización y análisis avanzado diseñado para inspeccionar interacciones entre estudiantes y un agente tutor multimodal de física (basado en un sistema RAG). 

La aplicación se conecta directamente a la API de **LangSmith** para extraer y procesar automáticamente las trazas de uso reales. Extrae inteligentemente la pregunta original del estudiante (ignorando inyecciones del sistema en el prompt), la respuesta generada por el agente y el contexto específico que fue recuperado de Qdrant.

La finalidad principal del proyecto no es solo observar de manera pasiva, sino **evaluar la calidad pedagógica, técnica y de recuperación** de las interacciones para retroalimentar al sistema central.

---

## 🎯 Objetivo Final: Integración con DSPy GEPA

La meta definitiva de este entorno de análisis es actuar como el motor de curación de datos para el ecosistema. Al analizar y puntuar la calidad de las respuestas del tutor y la pertinencia de los documentos recuperados por el Vector Store, podemos construir un **dataset normativo de alta fidelidad**.

Este dataset (compuesto de pares de preguntas y respuestas ideales) será inyectado en el pipeline de **DSPy GEPA** (Generative Prompt Optimization) que se está desarrollando, permitiendo dos grandes avances:

1. **Ajuste y Sintonía del Vector Store (Qdrant):**  
   Al identificar sistemáticamente cuándo el contexto recuperado aporta "ruido" o carece de la teoría fundamental (gracias al análisis automatizado), se puede afinar la estrategia de *chunking*, re-ranking y umbrales de búsqueda en la base de datos vectorial.
   
2. **Optimización de Prompts mediante GEPA:**  
   Usando el historial depurado y calificado como *ground-truth* (datos de verdad terreno), el optimizador DSPy GEPA ajustará las firmas y restricciones del modelo subyacente. El objetivo es que la IA asimile a la perfección un "perfil de profesor normativo": dominando el estilo socrático, evitando respuestas directas prematuras y guiando al alumno al descubrimiento de los principios físicos (ej. Leyes de Newton, Bernoulli).

---

## 🏗️ Arquitectura y Stack Tecnológico

El dashboard es un monorepo que se ejecuta de forma concurrente, dividido en dos componentes:

- **Frontend (Astro.js + Vanilla CSS):** Una interfaz de usuario rápida e interactiva (SSR/CSR) que renderiza las tarjetas de interacción. Procesa Markdown complejo y maneja los estados de carga de manera fluida.
- **Backend (FastAPI + LangGraph):** Una API en Python encargada de la lógica pesada. Se comunica con LangSmith para agrupar e inferir contextos, y orquesta los agentes de Inteligencia Artificial (utilizando **Gemini 2.5 Flash**) para realizar las evaluaciones con manejo automático de errores por límites de cuota (Rate Limiting HTTP 429).

---

## 🧠 Flujo del Pipeline de Agentes Analistas (LangGraph)

Para garantizar una evaluación rigurosa sin intervención humana constante, el dashboard incluye una función de **"✨ Análisis con IA"** en cada tarjeta. Al activarse, se desencadena un grafo de procesamiento inteligente basado en **LangGraph**, compuesto por cuatro nodos (agentes especialistas) que se ejecutan de manera colaborativa:

### 1. Agente Analista de la Pregunta (`question_analyst`)
- **Misión:** Evaluar la claridad y calidad de la duda del estudiante.
- **Acción:** Identifica qué conceptos físicos intenta comprender el alumno. Determina si la pregunta está bien formulada, si el estudiante demuestra una base teórica, o si simplemente está pidiendo una resolución directa (o acciones fuera de contexto como generación de imágenes).

### 2. Agente Analista de la Respuesta (`response_analyst`)
- **Misión:** Auditar la actuación pedagógica del tutor IA.
- **Acción:** Revisa la salida del agente frente a la duda planteada. Evalúa si la respuesta fue adecuada para un entorno académico, si el agente utilizó correctamente el método socrático de indagación, y detecta posibles imprecisiones o "alucinaciones" físicas.

### 3. Agente Analista de Recuperación (`retrieval_analyst`)
- **Misión:** Calificar el desempeño del sistema RAG.
- **Acción:** Compara la duda del alumno exclusivamente con los fragmentos de texto devueltos por **Qdrant**. Determina si la recuperación fue relevante o si hubo inyección de "ruido" (información que confunde al LLM) y marca qué teorías clave faltaron para resolver el problema.

### 4. Agente Sintetizador de Mejoras (`improvement_synthesizer`)
- **Misión:** Generar *insights* accionables para los desarrolladores.
- **Acción:** Reúne las conclusiones de los tres analistas anteriores y redacta de 3 a 5 recomendaciones técnicas o pedagógicas precisas (ej. "La IA debe abstenerse de resolver la ecuación de energía cinética en el primer paso", "Mejorar la indexación del Tema 9: Hidrodinámica en Qdrant").

*Este flujo de 4 fases asegura que cada dimensión del sistema (Usuario, Generación, Recuperación) se someta a auditoría crítica, preparando el terreno para el reentrenamiento automático con DSPy.*

---

## ⚙️ Instalación y Ejecución

1. **Clonar o situarse en el directorio del proyecto:**
   Asegúrate de estar en `langsmith-qa-dashboard`.

2. **Configurar las credenciales en el Backend:**
   Copiar o crear el archivo `backend/.env` con las siguientes llaves:
   ```env
   LANGSMITH_API_KEY=tu_api_key_de_langsmith
   LANGSMITH_PROJECT=tu_proyecto_de_trazas
   GOOGLE_API_KEY=tu_clave_de_google_gemini
   ```

3. **Instalar dependencias e iniciar:**
   Desde la raíz del proyecto, instala todos los paquetes (node y uv) y lanza los servidores de forma simultánea:
   ```bash
   npm install
   npm run dev
   ```

La aplicación estará disponible de forma local en `http://localhost:4321`.
