# AutoSage AI — Judge-facing pitch kit

## 20-second pitch

**AutoSage is an evidence-guided vehicle troubleshooting agent for two-wheelers and four-wheelers.** Instead of giving a generic chatbot answer, it plans a domain tool call, retrieves supporting technical knowledge, and generates a structured action guide with a deterministic safety gate.

## The hook

“Most people can describe that their vehicle feels wrong. They usually cannot translate that symptom into the right technical next step. AutoSage bridges that gap.”

## What makes it more than RAG

1. **LangChain orchestration:** prompt templates, chat-model adapters, output parsing and typed tools provide the agent workflow.
2. **Agentic planning:** the LangChain planner decides whether a specialized vehicle tool is needed.
3. **Real tool execution:** OBD code lookup, safety gate, system triage, maintenance check, or parts-action planner exposed as LangChain tools.
4. **Evidence grounding:** the LangChain Chroma retrieval layer feeds the final answer with retrieved research chunks and visible sources.
5. **Domain guardrail:** safety-sensitive patterns are handled by deterministic Python logic before the LLM writes the final response.
6. **Auditable UX:** the judge can open “Agent tool call” and see what actually happened.
7. **Action after diagnosis:** the product continues into a structured service request, mechanic acceptance, ETA/deal setting and customer confirmation.

## Best 3-minute live query

> My 2022 petrol Hyundai i20 shows P0420, the check-engine light is on, and the car feels mostly normal. What should I check, and do I need to replace anything?

### What to point at on screen

**1. Tool call**

`obd_code_lookup(P0420)`

Explain: “The agent recognized that a diagnostic code is present and called a specialized lookup tool.”

**2. RAG**

Show retrieved OBD/EPA and manufacturer guidance.

Explain: “The tool gives structured facts; RAG provides contextual evidence for the final explanation.”

**3. Design choice**

Say:

> “We deliberately separated safety-sensitive logic from generation. If a symptom looks dangerous, deterministic Python can escalate it even if the language model would otherwise produce a fluent but unsafe answer.”

**4. Final response**

Point out that AutoSage says the code narrows a system and does not automatically prove that the catalytic converter must be replaced.

## Likely judge questions

### Why RAG instead of just an LLM?

Vehicle guidance needs traceable evidence. RAG gives us a small, auditable domain corpus and lets the final answer point back to sources.

### Why not let the LLM do everything?

The LLM handles language and synthesis, but high-risk rules and structured lookups are more predictable as deterministic tools.

### Why Streamlit?

For a hackathon, Streamlit gives a very fast path from Python code to a shareable application. The UI is heavily customized with HTML/CSS, while the architecture remains modular enough to move to a React frontend later.

### How does it scale?

The next layer is a remote vector database, model-specific manual ingestion, an external cost/service API, and optional Bluetooth OBD-II input.

### How do you prevent hallucinated part replacement?

The system prompt explicitly requires “inspect/test before replacement”, the tool outputs are fed into the final prompt, and answers distinguish likely causes from confirmed faults.

## Service Dispatch — the show-stopper sequence

After the diagnosis, the demo can continue without leaving AutoSage:

```text
HIGH-RISK / SERVICE RECOMMENDED
        ↓
User shares location
        ↓
Nearby repair shops on map
        ↓
User selects a shop
        ↓
Appointment request includes:
  • vehicle
  • reported problem
  • AutoSage assessment
  • location
  • customer contact
  • preferred slot
        ↓
Mechanic Portal
        ↓
Accept / Reject
        ↓
ETA + mechanic phone + estimated deal
        ↓
🎉 Confirmation card for user
```

**Demo line:** “We don't stop at diagnosis. AutoSage turns a problem into an actionable service request, lets the shop owner accept the job, set an ETA and deal, then gives the customer a confirmed contact.”

The current hackathon build uses a local SQLite request store and a user-triggered WhatsApp link when a public shop number is available. It does not claim to silently message real businesses. A production version should use authenticated mechanic accounts and a verified business-messaging/dispatch provider.

## Specific service tools

The location/service handoff also exposes LangChain tool wrappers for `geocode_location`, `find_nearby_mechanics`, service-request lookup/status, and customer confirmation generation.


- `create_service_request()`
- LangChain service tools: `service_request_lookup`, `service_request_status`, `service_request_confirmation`
- `get_service_request()` / `get_service_requests()`
- `update_service_request()`
- `build_mechanic_notification_message()`
- `build_customer_confirmation_message()`
- `build_whatsapp_link()`

## Stretch feature order

### Already in the demo

- Hindi + English + Gujarati output
- Whisper voice input
- Nearby mechanic search
- Select-shop appointment request
- Mechanic Portal acceptance flow
- ETA + estimated deal
- Customer confirmation + Call / WhatsApp handoff

### Add next

- Dashboard-warning image upload
- Mechanic-ready PDF report
- Saved vehicle profile

### Add after that

- Live workshop/service lookup
- Cost ranges from a controlled regional dataset
- Personalized maintenance reminders

### Ambitious

- Bluetooth OBD-II adapter data
- Time-series sensor dashboard
- Manufacturer-specific manuals and service procedures

## Hinglish demo angle

AutoSage accepts natural Hinglish / Romanized Hindi, so the live demo can use a realistic Indian-user query without switching interfaces:

> **Meri 2022 petrol car mein P0420 aa raha hai, check-engine light on hai aur gaadi normally chal rahi hai. Kya check karna chahiye?**

The language selector supports Auto, Hinglish, English, and Hindi.

## New Product Feature: Safety-to-Service Handoff + Appointment Dispatch

AutoSage does not stop at “see a mechanic.” When a deterministic safety tool marks a problem HIGH/CRITICAL, the UI explicitly offers a **Nearby Mechanics** flow. Location permission is optional. The user can share device coordinates or enter an area/city/pincode; `find_nearby_mechanics` searches OpenStreetMap repair POIs and distance-sorts them.

The user selects a shop and submits a service request. The request carries the vehicle, reported problem, AutoSage assessment, contact details, preferred slot and the explicitly shared service location. The Mechanic Portal then lets the shop/service-desk side accept or reject it and set an ETA, mechanic name/phone, and estimated deal. The user's status card becomes:

> 🎉 **Congratulations! Your mechanic appointment is CONFIRMED.**

**Demo line:** “We designed the handoff so the AI doesn't secretly track location. The user decides whether to share location and appointment details; only then does AutoSage create the service request. The mechanic side controls acceptance, ETA and the deal.”
