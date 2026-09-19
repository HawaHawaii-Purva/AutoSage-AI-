---
title: AutoSage AI
emoji: 🚘
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
---

# AutoSage AI 🚘

**Evidence-guided vehicle troubleshooting for 2-wheelers and 4-wheelers.**

AutoSage is a hackathon-ready AI vehicle assistant that combines:

- **RAG** over a curated vehicle safety/troubleshooting corpus
- **LangChain orchestration** for prompt templates, chat-model adapters, output parsing and structured tools
- **LLM agent planning**
- **Deterministic domain tools** for OBD codes, safety gating, symptom triage, maintenance and parts-action planning
- **A safety-first response policy** that distinguishes likely causes from confirmed diagnosis
- **English + Hinglish + Hindi + Gujarati support** with automatic script/language detection
- **Multilingual Whisper voice input** for spoken vehicle problems in English, Hindi, Gujarati, and Hinglish
- **Streamlit UI** with an auditable agent trace and research-source view

## Architecture

The implementation is intentionally LangChain-first.

**Pinned LangChain stack in this build:** `langchain==1.3.14`, `langchain-core==1.5.3`, `langchain-ollama==1.1.0`, `langchain-chroma==1.1.0`, `langchain-huggingface==1.2.2`. These are current 2026 releases selected from PyPI so the demo environment is reproducible. Local Ollama and optional Hugging Face inference are exposed through LangChain chat models; diagnostic functions are registered as LangChain tools; the RAG layer uses LangChain Documents + the LangChain Chroma integration; the final response is a composable LangChain prompt/model/parser chain. Voice transcription (Whisper), OpenStreetMap HTTP calls and Streamlit UI remain native integrations because LangChain does not add useful abstraction to those components.


```text
User
  ↓
Streamlit UI
  ├── 🎙️ Multilingual Whisper voice transcription
  └── Text input (English / Hindi / Hinglish / Gujarati)
  ↓
LangChain planner chain (ChatPromptTemplate → ChatModel → StrOutputParser)
  ↓
LangChain StructuredTools + LangChain Chroma RAG
  ├── OBD code lookup
  ├── Safety gate
  ├── Symptom/system triage
  ├── Maintenance check
  └── Parts action plan
  ↓
Structured troubleshooting guide + sources + trace
  ↓
Professional-service handoff (user controlled)
  ├── 📍 Geolocation / area lookup
  ├── 🔧 Nearby mechanic search (OpenStreetMap)
  ├── 📝 Appointment request + mechanic brief
  ├── 🛠️ Mechanic Portal: accept/reject + ETA + deal
  └── 🎉 Customer confirmation + Call / WhatsApp
```

## Voice + Multilingual UX

AutoSage includes a microphone input powered by **faster-whisper**. The multilingual Whisper model can auto-detect spoken English, Hindi, Gujarati, and mixed/Hinglish speech, then pass the transcription into the same agent + RAG pipeline used for typed queries.

Voice model settings can be controlled with environment variables:

```text
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

The default model is `small`. The first voice transcription downloads the model, so the first run can take longer. For a faster/lighter demo, change `transcribe_audio(audio, voice_language, model_size="base")` in `app.py` or make the model size configurable.

### Gujarati support

The response-language selector now includes **Gujarati**. Auto mode recognizes Gujarati Unicode text and asks the LLM to answer in Gujarati script while retaining technical terms such as OBD-II, ABS, P0420, brake pad, and oxygen sensor in English where useful.


AutoSage accepts natural-language input in English, Hindi, and Hinglish (Romanized Hindi). The UI offers **Auto / Hinglish / English / Hindi / Gujarati** response modes. In Auto mode, a lightweight heuristic detects common Roman-Hindi markers and tells the LLM to answer in a matching Hinglish style. This is not a translation layer: it preserves technical terms such as OBD-II, P0420, ABS, brake pad, and oxygen sensor.

Example input:

> **Meri 2022 petrol car mein P0420 aa raha hai, check-engine light on hai aur gaadi normally chal rahi hai. Kya check karna chahiye?**

## Deliberate design decision

**Safety-sensitive logic is deterministic.** The LLM can interpret and explain a vehicle symptom, but a plain-Python safety tool is used to flag obvious high-risk patterns. The final model prompt is required to respect the tool result and escalate rather than suggest risky experimentation.

## Local setup (VS Code)

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Make sure Ollama is running

Example model:

```bash
ollama pull llama3.2:3b
ollama list
```

### 4. Create `.env`

Copy `.env.example` to `.env`.

Default local settings:

```text
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

### 5. Run

```bash
streamlit run app.py
```

### LangChain observability

The app works without LangSmith. For judge/demo tracing, set `LANGSMITH_TRACING=true`, add a `LANGSMITH_API_KEY`, and optionally set `LANGSMITH_PROJECT=AutoSage-AI`. LangChain will trace the model/chain runs without changing the user-facing flow.

## LangChain implementation map

- `llm.py` → `ChatOllama` / `ChatHuggingFace`
- `agent.py` → `ChatPromptTemplate` + LangChain chat model + `StrOutputParser`
- `tools.py` → typed `@tool` functions collected in `LANGCHAIN_TOOLS`
- `rag_engine.py` → LangChain `Document` + `Chroma` vector store
- `service_dispatch.py` → LangChain service-status/confirmation tools
- `voice.py` → faster-whisper (kept native because Whisper is the transcription engine)
- `app.py` → Streamlit UI and explicit human-in-control location/service handoff

## Best demo query

Use a real tool call that judges can see:

> **My 2022 petrol Hyundai i20 shows P0420, the check-engine light is on, and the car feels mostly normal. What should I check, and do I need to replace anything?**

Expected path:

```text
LLM planner
   ↓
obd_code_lookup(P0420)
   ↓
Chroma retrieves OBD + manufacturer guidance
   ↓
LLM synthesizes diagnosis guide
   ↓
UI shows tool trace + sources
```

## 3-minute live demo script

**0:00–0:25 — Problem**

“Vehicle problems are often described in natural language, while useful troubleshooting information is scattered across manuals and technical resources. AutoSage turns that into an evidence-guided workflow.”

**0:25–0:50 — Show UI**

Select four-wheeler, enter vehicle details, enter the P0420 query.

**0:50–1:30 — Run diagnosis**

Point to the agent trace: the planner selected `obd_code_lookup`, the tool returned the generic meaning and checks, and RAG retrieved supporting material.

**1:30–2:15 — Explain result**

Show that the answer does not blindly say “replace catalytic converter”. It recommends diagnosis first, lists checks, and explains that a DTC narrows the system but does not prove a failed part.

**2:15–2:45 — Design decision**

“We deliberately made the safety gate deterministic. The LLM handles language, but high-risk vehicle patterns are handled by testable Python logic before the final response.”

**2:15–2:35 — Service handoff**

Open **Nearby Mechanics**, share/enter location, select a shop and submit an appointment request.

**2:35–2:50 — Mechanic Portal**

Switch to **Mechanic Portal**, open the request and accept it with a mechanic name, phone, ETA and estimated deal.

**2:50–3:00 — Show the wow moment**

Return to the status card: **“🎉 Congratulations! Your mechanic appointment is CONFIRMED.”** Show the ETA, mechanic contact, deal note and WhatsApp/Call actions.


## Safety-to-Service Dispatch 🚗 → 🛠️

AutoSage now has an end-to-end hackathon workflow after diagnosis:

```text
Diagnosis
   ↓
Explicit location consent
   ↓
find_nearby_mechanics()
   ↓
User selects a repair shop
   ↓
create_service_request()
   ↓
Mechanic Portal
   ↓
Owner accepts/rejects
   ├── ETA
   ├── mechanic name + phone
   ├── estimated service/deal fee
   └── appointment note
   ↓
🎉 Appointment confirmed
   ↓
Call / WhatsApp the mechanic
```

### What gets sent to the mechanic

The appointment request automatically packages the information needed for a service handoff: customer contact, vehicle details, reported symptom, the AutoSage assessment, diagnostic tool used, preferred slot, and the service location the user explicitly agreed to share.

### Mechanic Portal

The **🛠️ Mechanic Portal** tab acts as the shop owner/service desk for the hackathon demo. An owner can open an incoming request, review the problem, then **accept/reject**, set the mechanic's ETA, enter an estimated fee/deal note, and write a customer-facing message.

When accepted, the customer sees:

> 🎉 **Congratulations! Your mechanic appointment is CONFIRMED.** The assigned mechanic has agreed to take the request and will reach you within the confirmed ETA, with their contact number and deal note shown on the confirmation card.

### Explicit third-party handoff

Where a shop's public phone number is available in OpenStreetMap, AutoSage can open a pre-filled WhatsApp message for the user. The app does **not** pretend to silently send a third-party message. Automatic owner notifications in a production system should use an authenticated backend plus a verified WhatsApp/SMS/dispatch provider.

### Deterministic tools

The service layer adds these concrete tools/functions:

- `create_service_request()` — create a pending appointment request.
- `get_service_request()` / `get_service_requests()` — read appointment status.
- `update_service_request()` — mechanic owner changes status, ETA and deal details.
- `build_mechanic_notification_message()` — prepare a mechanic-ready problem brief.
- `build_customer_confirmation_message()` — prepare the confirmation text.
- `build_whatsapp_link()` — create a user-triggered WhatsApp handoff link.

The hackathon implementation uses **SQLite locally** so the state transition is deterministic and easy to demo. For production, replace the local database with a hosted service, add authenticated mechanic accounts, and use real-time notifications.

## Cloud deployment: Streamlit Community Cloud

1. Push this project to a **public GitHub repository**.
2. Go to Streamlit Community Cloud and connect GitHub.
3. Create an app from the repository and select `app.py`.
4. Add the secret:

```toml
LLM_PROVIDER = "hf"
HF_TOKEN = "hf_your_token_here"
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
```

Do **not** commit the Hugging Face token to GitHub.

Community Cloud can deploy directly from a GitHub repository, and repository changes can trigger updated deployments. See the official Streamlit deployment documentation.

## GitHub commands

```bash
git init
git add .
git commit -m "Initial AutoSage AI hackathon build"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AutoSage-AI.git
git push -u origin main
```

## Recommended stretch features

### Tier 1 — high value / low complexity

- Voice symptom input
- Saved vehicle profiles
- Conversation history
- Multilingual Hindi/English mode
- “Mechanic-ready report” export

### Tier 2 — wow factor

- Upload a dashboard-warning photo
- Upload a visible part/tyre/brake photo
- Interactive vehicle health timeline
- Personalized maintenance reminders
- Service-cost estimates from a controlled dataset

### Tier 3 — production dispatch

- Hosted realtime appointment backend
- Authenticated mechanic/shop accounts
- Verified WhatsApp/SMS dispatch integration
- Bluetooth OBD-II adapter integration
- Live scan-data dashboard
- Manufacturer/model-specific manual retrieval

## Safety boundary

AutoSage is a prototype decision-support system. It should not be presented as a certified mechanic or as a substitute for an owner's manual, qualified technician, diagnostic equipment, or emergency/roadside assistance.

## Research references

- NHTSA Tire Safety: https://www.nhtsa.gov/vehicle-safety/tires
- Motorcycle Safety Foundation T-CLOCS: https://msf-usa.org/documents/library/t-clocs-pre-ride-inspection-checklist/
- U.S. EPA OBD resources: https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100LW9G.TXT
- Hyundai Owner's Manuals: https://ownersmanual.hyundai.com/
- Gradio / Hugging Face Spaces docs: https://huggingface.co/docs/hub/spaces-overview
- Streamlit Community Cloud docs: https://docs.streamlit.io/deploy/streamlit-community-cloud

## Nearby Mechanic Finder

When AutoSage detects a higher-risk symptom, it can recommend professional inspection. The user can then open **Nearby Mechanics** and either:

1. Share browser/device location (optional), or
2. Enter an area, city, or pincode.

The `find_nearby_mechanics` tool queries OpenStreetMap's Overpass API for nearby vehicle-repair POIs, sorts them by distance, and displays them on a map. The user then selects one shop and can create a structured appointment request.

The location flow is explicit and user-controlled. Appointment location/contact details are stored only as part of the request the user deliberately submits. The demo uses local SQLite for request state and does not silently contact businesses.

This uses the public OpenStreetMap Nominatim geocoder only for a user-triggered area lookup, with an identifiable User-Agent and a switchable endpoint. Keep traffic low and cache/replace the service if your usage grows. See the OpenStreetMap Foundation Nominatim policy: https://operations.osmfoundation.org/policies/nominatim/
