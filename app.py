from __future__ import annotations

import os
import time
import html
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from streamlit_geolocation import streamlit_geolocation
from agent import analyze
from tools import find_nearby_mechanics, geocode_location
from voice import transcribe_audio
from service_dispatch import (
    build_customer_confirmation_message,
    build_mechanic_notification_message,
    build_whatsapp_link,
    create_service_request,
    get_service_request,
    get_service_requests,
    update_service_request,
)

# Streamlit Cloud secrets are copied into the environment for local/cloud parity.
try:
    for _key in (
        "LLM_PROVIDER", "HF_TOKEN", "HF_MODEL", "OLLAMA_MODEL", "OLLAMA_BASE_URL",
        "OSM_USER_AGENT", "OSM_NOMINATIM_URL", "OVERPASS_URL", "SERVICE_DB_PATH",
        "LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT",
    ):
        if _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

st.set_page_config(
    page_title="AutoSage AI — Vehicle Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#070914; --panel:#0d1222; --panel2:#11172b; --line:rgba(255,255,255,.10); --text:#f7f8ff; --muted:#99a2bd; --violet:#8b6cff; --cyan:#53e7ff; --green:#4fe0a4; --orange:#ffb454; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(circle at 15% 5%, rgba(139,108,255,.18), transparent 30%), radial-gradient(circle at 90% 0%, rgba(83,231,255,.10), transparent 25%), var(--bg); }
.block-container { max-width:1300px; padding-top:2rem; padding-bottom:4rem; }
.hero { padding:34px 36px; border:1px solid var(--line); border-radius:28px; background:linear-gradient(135deg,rgba(20,26,50,.92),rgba(9,13,27,.84)); box-shadow:0 24px 80px rgba(0,0,0,.35); position:relative; overflow:hidden; }
.hero:after { content:''; position:absolute; width:360px; height:360px; right:-120px; top:-180px; border-radius:50%; border:1px solid rgba(83,231,255,.20); box-shadow:0 0 0 40px rgba(83,231,255,.03),0 0 0 80px rgba(83,231,255,.02); }
.eyebrow { color:var(--cyan); font-size:.78rem; letter-spacing:.16em; text-transform:uppercase; font-weight:700; }
.hero h1 { font-family:'Space Grotesk'; font-size:3.3rem; line-height:1.0; margin:.45rem 0 .8rem; letter-spacing:-.045em; }
.hero p { color:var(--muted); font-size:1.05rem; max-width:760px; }
.pill { display:inline-flex; align-items:center; gap:8px; padding:7px 11px; border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.04); color:#dfe4f5; font-size:.82rem; margin:5px 6px 0 0; }
.card { border:1px solid var(--line); border-radius:20px; padding:22px; background:linear-gradient(180deg,rgba(18,24,46,.86),rgba(11,16,31,.88)); }
.card h3 { font-family:'Space Grotesk'; margin-top:0; }
.trace { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.83rem; color:#d7dcf1; background:#070a14; border:1px solid var(--line); border-radius:14px; padding:15px; }
.source { border:1px solid var(--line); border-radius:15px; padding:14px 16px; margin:8px 0; background:rgba(255,255,255,.025); }
.source a { color:var(--cyan); text-decoration:none; }
.small { color:var(--muted); font-size:.86rem; }
.warning { border-left:3px solid var(--orange); background:rgba(255,180,84,.08); padding:13px 16px; border-radius:10px; }
.good { border-left:3px solid var(--green); background:rgba(79,224,164,.08); padding:13px 16px; border-radius:10px; }
.danger { border-left:3px solid #ff6678; background:rgba(255,102,120,.08); padding:13px 16px; border-radius:10px; }
.mechanic-card { border:1px solid var(--line); border-radius:18px; padding:18px; background:linear-gradient(180deg,rgba(16,22,42,.92),rgba(8,12,25,.94)); margin-bottom:12px; }
.mechanic-title { font-family:'Space Grotesk'; font-size:1.03rem; font-weight:700; color:#f7f8ff; }
.mechanic-meta { color:var(--muted); font-size:.83rem; margin-top:4px; }
.badge { display:inline-block; margin-top:9px; padding:5px 8px; border-radius:999px; border:1px solid var(--line); background:rgba(83,231,255,.06); color:#dffbff; font-size:.74rem; }
button[kind="primary"] { border-radius:14px !important; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0a0f1d 0%,#070914 100%); border-right:1px solid var(--line); }
label, .stMarkdown, .stText, .stCaption { color:#e9edfa; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result = None
if "mechanics" not in st.session_state:
    st.session_state.mechanics = None
if "location" not in st.session_state:
    st.session_state.location = None
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = uuid.uuid4().hex
if "last_problem" not in st.session_state:
    st.session_state.last_problem = ""
if "selected_request_id" not in st.session_state:
    st.session_state.selected_request_id = None

with st.sidebar:
    st.markdown("## 🚘 AutoSage AI")
    st.caption("Vehicle troubleshooting intelligence")
    st.divider()
    vehicle_type_label = st.radio("Vehicle", ["🏍️ Two-wheeler", "🚗 Four-wheeler"], index=1)
    vehicle_type = "two_wheeler" if "Two" in vehicle_type_label else "four_wheeler"
    make = st.text_input("Brand", placeholder="e.g. Hyundai")
    model = st.text_input("Model", placeholder="e.g. i20")
    year = st.text_input("Year", placeholder="e.g. 2022")
    fuel = st.selectbox("Fuel / power", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid", "Unknown"])
    language_mode = st.selectbox(
        "Response language",
        ["Auto", "Hinglish", "English", "Hindi", "Gujarati"],
        help="Auto mode detects English, Hindi, Gujarati, and common Hinglish/Roman-Hindi patterns.",
    )
    st.divider()
    provider = os.getenv("LLM_PROVIDER", "ollama").upper()
    st.markdown(f"**LLM:** `{provider}`")
    st.caption("Local: Ollama • Deploy: Hugging Face Inference")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-powered vehicle intelligence</div>
  <h1>Know what's wrong.<br>Know what to do next.</h1>
  <p>AutoSage combines retrieval-grounded vehicle knowledge, deterministic safety tools and an LLM agent to turn everyday vehicle symptoms into a structured troubleshooting plan — and can connect you to nearby repair shops when professional inspection is recommended.</p>
  <div>
    <span class="pill">🏍️ 2-Wheeler</span><span class="pill">🚗 4-Wheeler</span><span class="pill">🎙️ Whisper Voice</span><span class="pill">🌐 Gujarati</span><span class="pill">🧠 RAG</span><span class="pill">🛠️ Tools</span><span class="pill">🤖 Agent</span><span class="pill">🧩 LangChain</span><span class="pill">📍 Nearby Mechanics</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Diagnose", "📍 Nearby Mechanics", "🧠 System Trace", "📚 Research Base", "🛠️ Mechanic Portal"])

with tab1:
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown('<div class="card"><h3>Describe the symptom</h3><div class="small">Write it naturally. Add a dashboard code, warning light, sound, timing, or driving condition when you know it.</div></div>', unsafe_allow_html=True)
        st.write("")
        default_query = st.session_state.get("prefill", "")
        query = st.text_area(
            "Describe the symptom",
            value=default_query,
            placeholder="Example: Meri 2022 petrol car mein P0420 aa raha hai, check-engine light on hai aur gaadi normally chal rahi hai. Kya check karna chahiye?",
            height=170,
            label_visibility="collapsed",
        )
        st.caption("💬 Hinglish is supported — e.g. *Meri bike start nahi ho rahi, self maarne pe click ki awaaz aa rahi hai. Kya check karu?*")
        st.markdown("### 🎙️ Voice input")
        st.caption("Speak in English, Hindi, Gujarati, or Hinglish. AutoSage uses a multilingual Whisper model to transcribe your voice before diagnosis.")
        voice_language = st.selectbox("Voice language", ["Auto-detect", "English", "Hindi", "Gujarati"], key="voice_language")
        audio = st.audio_input("Record your vehicle problem", key="vehicle_voice")
        if audio is not None:
            if st.button("📝 Transcribe voice", use_container_width=True):
                with st.spinner("Whisper is listening and transcribing..."):
                    voice_result = transcribe_audio(audio, voice_language)
                if voice_result["ok"]:
                    st.session_state.prefill = voice_result["text"]
                    detected = voice_result.get("language") or "unknown"
                    st.success(f"Transcribed successfully • detected language: {detected}")
                    st.rerun()
                else:
                    st.error(voice_result["error"])
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⚡ No-start", use_container_width=True):
                st.session_state.prefill = "Meri vehicle start nahi ho rahi. Self press karne pe clicking ki awaaz aa rahi hai."
                st.rerun()
        with c2:
            if st.button("🟠 Warning light", use_container_width=True):
                st.session_state.prefill = "Meri car mein check-engine light on hai aur P0420 aa raha hai. Iska kya matlab hai aur mujhe kya check karna chahiye?"
                st.rerun()
        with c3:
            if st.button("🔴 Brake issue", use_container_width=True):
                st.session_state.prefill = "Meri car ke brakes se grinding ki awaaz aa rahi hai aur braking pehle se weak lag rahi hai."
                st.rerun()
        st.write("")
        if st.button("🚀 RUN SMART DIAGNOSIS", type="primary", use_container_width=True):
            if not query.strip():
                st.warning("Describe the vehicle problem first.")
            elif not make.strip() or not model.strip():
                st.warning("Add at least the vehicle brand and model in the sidebar.")
            else:
                with st.spinner("Agent is planning → calling tools → retrieving evidence → synthesizing..."):
                    st.session_state.result = analyze(vehicle_type, make, model, year or "Unknown year", fuel, query.strip(), language_mode)
                    st.session_state.last_problem = query.strip()
                    st.session_state.prefill = ""
                    st.session_state.mechanics = None
    with right:
        st.markdown("""
        <div class="card">
        <h3>Designed for real-world uncertainty</h3>
        <p class="small">AutoSage separates <b>evidence</b>, <b>tool outputs</b> and <b>LLM explanation</b>. It never treats a symptom as proof that a particular part is bad.</p>
        <div class="good"><b>Safety-first gate</b><br>Brake, steering, fuel-leak, severe overheating and other red-flag symptoms are escalated instead of receiving risky DIY advice.</div>
        <br>
        <div class="good"><b>Mechanic handoff</b><br>When a high-risk issue is detected, the user can explicitly share their location to find nearby repair shops.</div>
        <br>
        <div class="warning"><b>Privacy by design</b><br>AutoSage only uses location when the user chooses the location finder. The app itself does not save a location database.</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.result:
        result = st.session_state.result
        st.write("")
        st.markdown('<div class="card"><h3>Diagnostic guidance</h3></div>', unsafe_allow_html=True)
        st.markdown(result["answer"])

        if result.get("mechanic_recommended"):
            st.markdown("""
            <div class="danger"><b>Professional inspection recommended.</b><br>
            This result contains a higher-risk symptom. Open the <b>Nearby Mechanics</b> tab to share your location and find repair shops in the area.</div>
            """, unsafe_allow_html=True)

        with st.expander("🔧 Agent tool call — show me the evidence", expanded=True):
            st.markdown(f'<div class="trace">FRAMEWORK: LangChain<br>CHAIN: ChatPromptTemplate → ChatModel → StrOutputParser<br><br>TOOL: {html.escape(str(result["tool"]))}<br>REASON: {html.escape(str(result["planner_reason"] or "Deterministic fallback"))}<br><br>RESULT:<br>{html.escape(str(result["tool_result"]))}</div>', unsafe_allow_html=True)

        st.caption(f"Detected input: {result.get('detected_language', 'unknown')} • Response: {result.get('response_language', 'Auto')}")
        st.markdown("### Retrieved research")
        for d in result["evidence"]:
            meta = d["metadata"]
            st.markdown(f'<div class="source"><b>{html.escape(str(meta.get("title","Source")))}</b><br><span class="small">{html.escape(str(meta.get("source_name","Unknown source")))}</span><br><a href="{html.escape(str(meta.get("source_url","#")))}" target="_blank">Open source</a></div>', unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="card">
    <h3>📍 Find → Select → Request → Confirm</h3>
    <p class="small">Share your location, choose a nearby shop, send the mechanic your vehicle problem, and wait for the shop owner to accept the request and set an ETA/deal.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1.0, 1.15], gap="large")
    with left:
        st.markdown("### 1. Share your service location")
        location_mode = st.radio(
            "Location method",
            ["Use my device location", "Enter area / city / pincode"],
            label_visibility="collapsed",
            key="service_location_mode",
        )
        coords = None
        if location_mode == "Use my device location":
            st.info("Your browser will ask for permission. AutoSage uses the coordinates only for this mechanic-search and appointment session.")
            device_location = streamlit_geolocation()
            if isinstance(device_location, dict):
                direct_lat = device_location.get("latitude")
                direct_lon = device_location.get("longitude")
                nested = device_location.get("coords") or {}
                lat = direct_lat if direct_lat is not None else nested.get("latitude")
                lon = direct_lon if direct_lon is not None else nested.get("longitude")
                if lat is not None and lon is not None:
                    coords = {"latitude": float(lat), "longitude": float(lon), "label": "Your device location"}
                    st.session_state.location = coords
                elif device_location.get("error"):
                    st.warning("Location permission was not available. You can use the area/city option instead.")
            elif st.session_state.location:
                coords = st.session_state.location
        else:
            area = st.text_input("Area / city / pincode", placeholder="e.g. Ahmedabad 380015", key="service_area")
            if st.button("📍 Locate this area", use_container_width=True):
                if not area.strip():
                    st.warning("Enter an area, city or pincode first.")
                else:
                    last_call = st.session_state.get("last_geocode_call", 0.0)
                    wait = 1.1 - (time.time() - last_call)
                    if wait > 0:
                        time.sleep(wait)
                    with st.spinner("Finding the area on the map..."):
                        loc = geocode_location(area)
                    st.session_state.last_geocode_call = time.time()
                    if loc.get("ok"):
                        st.session_state.location = {
                            "latitude": loc["latitude"],
                            "longitude": loc["longitude"],
                            "label": loc.get("display_name", area),
                        }
                        st.success(f"Located: {loc.get('display_name', area)}")
                    else:
                        st.error(loc.get("error", "Could not locate that area."))
            coords = st.session_state.location

        radius_km = st.slider(
            "Search radius",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="The public Overpass search is limited to a modest radius for responsiveness.",
            key="service_radius",
        )

        if coords:
            st.success(f"📍 {coords.get('label','Location ready')}: {coords['latitude']:.5f}, {coords['longitude']:.5f}")
            if st.button("🔧 FIND NEARBY MECHANICS", type="primary", use_container_width=True):
                with st.spinner("Searching nearby repair shops..."):
                    st.session_state.mechanics = find_nearby_mechanics(
                        coords["latitude"], coords["longitude"], vehicle_type, int(radius_km * 1000)
                    )
        else:
            st.caption("Share your location or enter an area to activate the mechanic finder.")

    with right:
        result = st.session_state.get("mechanics")
        if result and result.get("ok"):
            rows = result.get("mechanics", [])
            if rows:
                import pandas as pd
                map_rows = [{"lat": r["latitude"], "lon": r["longitude"], "name": r["name"]} for r in rows]
                st.map(pd.DataFrame(map_rows), latitude="lat", longitude="lon", zoom=13, height=330)
                st.caption(f"Showing {len(rows)} nearby repair listings • {result.get('attribution','© OpenStreetMap contributors')}")
            else:
                st.info("No repair listings were found in this radius. Try a larger radius or a more central location.")
        elif result and not result.get("ok"):
            st.error(result.get("error", "Nearby mechanic search failed."))
        else:
            st.markdown("""
            <div class="card">
            <h3>Show-stopper flow</h3>
            <div class="trace">USER LOCATION<br>↓<br>find_nearby_mechanics()<br>↓<br>SELECT SHOP<br>↓<br>create_service_request()<br>↓<br>MECHANIC PORTAL<br>↓<br>ACCEPT + ETA + DEAL<br>↓<br>🎉 USER CONFIRMATION + CONTACT</div>
            </div>
            """, unsafe_allow_html=True)

    rows = st.session_state.get("mechanics", {}).get("mechanics", []) if st.session_state.get("mechanics") else []
    if rows:
        st.write("")
        st.markdown("### 2. Select a mechanic shop")
        options = list(range(len(rows)))
        selected_idx = st.selectbox(
            "Mechanic shop",
            options,
            format_func=lambda i: f"{rows[i]['name']} • {rows[i]['distance_km']:.2f} km • {rows[i]['specialization']}",
            key="selected_mechanic_idx",
        )
        selected_shop = rows[selected_idx]
        maps_url = f"https://www.google.com/maps/search/?api=1&query={selected_shop['latitude']},{selected_shop['longitude']}"
        st.markdown(f"""
        <div class="mechanic-card">
          <div class="mechanic-title">Selected: {html.escape(str(selected_shop['name']))}</div>
          <div class="mechanic-meta">{selected_shop['distance_km']:.2f} km away • {html.escape(str(selected_shop['specialization']))}</div>
          <span class="badge">{html.escape(str(selected_shop['compatibility']))}</span>
          <div class="small" style="margin-top:10px">{html.escape(str(selected_shop['address']))}</div>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("🗺️ Open shop in Maps", maps_url, use_container_width=True)

        if not st.session_state.get("last_problem"):
            st.warning("Run Smart Diagnosis first so AutoSage can attach the actual vehicle problem and diagnostic context to the mechanic request.")
        else:
            st.markdown("### 3. Send the appointment request")
            st.caption("The request package contains the vehicle details, reported problem, AI assessment, service location and preferred slot. The owner can then accept/reject and set the ETA/deal.")
            with st.form("appointment_request_form"):
                c1, c2 = st.columns(2)
                with c1:
                    customer_name = st.text_input("Your name *", placeholder="e.g. Purva")
                with c2:
                    customer_phone = st.text_input("Your phone *", placeholder="e.g. 9876543210")
                preferred_slot = st.selectbox("Preferred appointment", [
                    "As soon as possible",
                    "Within 30 minutes",
                    "Within 1 hour",
                    "Today evening",
                    "I will coordinate with the mechanic",
                ])
                location_label = coords.get("label", "Shared service location") if coords else "Location not shared"
                consent = st.checkbox(
                    "I agree to share my vehicle problem, contact details and the selected service location with the mechanic for this appointment.",
                    value=False,
                )
                submit_request = st.form_submit_button("🚀 REQUEST MECHANIC APPOINTMENT", type="primary", use_container_width=True)
                if submit_request:
                    if not customer_name.strip() or not customer_phone.strip():
                        st.error("Enter your name and phone number so the mechanic can contact you.")
                    elif not consent:
                        st.error("Please give explicit consent before sharing your details with the mechanic.")
                    else:
                        with st.spinner("Creating the service request and preparing the mechanic brief..."):
                            req = create_service_request(
                                user_session_id=st.session_state.session_id,
                                customer_name=customer_name,
                                customer_phone=customer_phone,
                                service_location_label=location_label,
                                service_lat=coords.get("latitude") if coords else None,
                                service_lon=coords.get("longitude") if coords else None,
                                vehicle=f"{year or 'Unknown year'} {make} {model} ({fuel})",
                                vehicle_type=vehicle_type,
                                problem=st.session_state.last_problem,
                                diagnosis=st.session_state.result.get("answer", "") if st.session_state.result else "",
                                diagnostic_tool=st.session_state.result.get("tool", "") if st.session_state.result else "",
                                shop=selected_shop,
                                preferred_slot=preferred_slot,
                            )
                        st.session_state.selected_request_id = req.get("id")
                        st.success(f"✅ Appointment request {req.get('id')} sent to {selected_shop['name']}.")
                        st.info("The mechanic portal can now review the problem and confirm the appointment with an ETA and deal.")
                        shop_phone = selected_shop.get("phone") or ""
                        if shop_phone:
                            wa_url = build_whatsapp_link(shop_phone, build_mechanic_notification_message(req))
                            if wa_url:
                                st.link_button("📲 Open WhatsApp with mechanic brief", wa_url, use_container_width=True)
                        else:
                            st.caption("This OpenStreetMap listing does not expose a public phone number, so the request is available through the Mechanic Portal instead of inventing contact details.")

    st.write("")
    st.markdown("### 4. Appointment status")
    st.caption("Refresh after the shop owner accepts/rejects the request. In a production deployment, a realtime backend/notification service would push this update automatically.")

    selected_id = st.session_state.get("selected_request_id")

    if selected_id:
        latest_request = get_service_request(selected_id)
        user_requests = [latest_request] if latest_request else []
    else:
        st.write("")
st.markdown("### 4. Appointment status")
st.caption(
    "Refresh after the shop owner accepts/rejects the request. "
    "In a production deployment, a realtime backend/notification service "
    "would push this update automatically."
)

# Always fetch the currently selected request directly.
selected_id = st.session_state.get("selected_request_id")
current_request = get_service_request(selected_id) if selected_id else None

# Fall back to this user's recent requests.
if current_request:
    user_requests = [current_request]
else:
    user_requests = get_service_requests(
        user_session_id=st.session_state.session_id,
        limit=20,
    )

if st.button("🔄 Refresh appointment status", use_container_width=False):
    st.rerun()

if not user_requests:
    st.caption("No appointment requests yet.")
else:
    for req in user_requests:
        status = str(req.get("status") or "").upper()

        if status == "ACCEPTED":
            eta = req.get("eta_minutes")
            mechanic = req.get("mechanic_name") or "Mechanic"
            phone = req.get("mechanic_phone") or "Contact not listed"

            st.markdown(
                f"""
                <div class="good">
                  <h3>🎉 Congratulations! Your mechanic appointment is CONFIRMED.</h3>
                  <b>{html.escape(str(mechanic))}</b>
                  from
                  <b>{html.escape(str(req.get('shop_name', 'Selected shop')))}</b>
                  has accepted your appointment and will reach you
                  <b>within {eta if eta is not None else 'the agreed'} minutes</b>.
                  <br><br>
                  📞 Mechanic:
                  <b>{html.escape(str(phone))}</b><br>
                  💰 Estimated deal:
                  <b>{html.escape(str(req.get('estimated_fee') or 'Not specified'))}</b><br>
                  📝 <b>Deal / appointment note:</b>
                  {html.escape(str(req.get('deal_note') or 'Appointment confirmed by the mechanic.'))}<br>
                  💬 <b>Message from mechanic:</b>
                  {html.escape(str(req.get('owner_message') or 'No additional message from the mechanic.'))}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if phone != "Contact not listed":
                c1, c2 = st.columns(2)

                with c1:
                    st.link_button(
                        "📞 Call mechanic",
                        f"tel:{phone}",
                        use_container_width=True,
                    )

                with c2:
                    wa = build_whatsapp_link(
                        phone,
                        build_customer_confirmation_message(req),
                    )
                    if wa:
                        st.link_button(
                            "💬 WhatsApp mechanic",
                            wa,
                            use_container_width=True,
                        )

        elif status == "REJECTED":
            st.error(
                f"Request {req.get('id')} was not accepted by "
                f"{req.get('shop_name')}."
            )

        elif status == "COMPLETED":
            st.success(
                f"Request {req.get('id')} is marked completed."
            )

        else:
            st.warning(
                f"⏳ Request {req.get('id')} is awaiting mechanic "
                f"confirmation from {req.get('shop_name')}."
            )
            if req.get("status") == "ACCEPTED":
                eta = req.get("eta_minutes")
                mechanic = req.get("mechanic_name") or "Mechanic"
                phone = req.get("mechanic_phone") or "Contact not listed"
                st.markdown(f"""
                <div class="good">
                  <b>🎉 Congratulations! Your mechanic appointment is CONFIRMED.</b><br><br>
                  <b>{html.escape(str(mechanic))}</b> from <b>{html.escape(str(req.get('shop_name','selected shop')))}</b> has agreed to take the appointment and will reach you <b>within {eta if eta is not None else 'the agreed'} minutes</b>.<br>
                  📞 Mechanic: <b>{html.escape(str(phone))}</b><br>
                  💰 Estimated deal: <b>{html.escape(str(req.get('estimated_fee') or 'Not specified'))}</b><br>
                  💬 <b>Message from mechanic:</b> {html.escape(str(req.get('owner_message') or 'No additional message from the mechanic.'))}
                  📝 {html.escape(str(req.get('deal_note') or req.get('owner_message') or 'Appointment confirmed by the mechanic.'))}
                </div>
                """, unsafe_allow_html=True)
                if phone != "Contact not listed":
                    cols = st.columns(2)
                    with cols[0]:
                        st.link_button("📞 Call mechanic", f"tel:{phone}", use_container_width=True)
                    with cols[1]:
                        wa = build_whatsapp_link(phone, build_customer_confirmation_message(req))
                        if wa:
                            st.link_button("💬 WhatsApp mechanic", wa, use_container_width=True)
            elif req.get("status") == "REJECTED":
                st.error(f"Request {req.get('id')} was not accepted by {req.get('shop_name')}.")
            elif req.get("status") == "COMPLETED":
                st.success(f"Request {req.get('id')} is marked completed.")
            else:
                st.warning(f"⏳ Request {req.get('id')} is awaiting mechanic confirmation from {req.get('shop_name')}.")

    if rows:
        st.caption("Mechanic listings and locations come from OpenStreetMap and may be incomplete or outdated. Confirm shop identity, availability and pricing before accepting any deal.")

with tab3:
    st.markdown("""
    <div class="card">
      <h3>What actually happens after you press Diagnose?</h3>
      <div class="trace">
      01  USER INPUT<br>
      &nbsp;&nbsp;&nbsp;vehicle type + make/model/year + symptom<br><br>
      02  AGENT PLANNER<br>
      &nbsp;&nbsp;&nbsp;LangChain prompt chain chooses a domain tool<br><br>
      03  TOOL EXECUTION<br>
      &nbsp;&nbsp;&nbsp;OBD lookup / safety gate / system triage / maintenance / parts plan<br><br>
      04  RAG RETRIEVAL<br>
      &nbsp;&nbsp;&nbsp;searches the curated vehicle knowledge corpus<br><br>
      05  LLM SYNTHESIS<br>
      &nbsp;&nbsp;&nbsp;turns evidence + tool result into a structured guide<br><br>
      06  HUMAN-IN-CONTROL HANDOFF<br>
      &nbsp;&nbsp;&nbsp;high-risk issue → explicit location sharing → nearby mechanic search → selected shop → service request → mechanic acceptance → ETA + deal → user confirmation
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("### Deliberate domain design decision")
    st.info("**The safety gate and location handoff are explicit, not hidden inside generation.** The LLM explains symptoms, but high-risk findings trigger a deterministic Python safety check. Only after the user chooses to share location does the mechanic-search tool run. This keeps safety and privacy decisions testable and user-controlled.")

    st.markdown("### Hackathon architecture")
    st.markdown("""
    - **RAG:** answers are grounded in a curated source corpus.
    - **LangChain:** prompt templates, chat-model adapters, output parsing, and typed tools form the orchestration layer.
    - **Agent:** chooses the appropriate domain tool through a LangChain planner chain.
    - **Tools:** deterministic code handles safety, OBD lookup, system triage, nearby-mechanic search, appointment creation, mechanic status updates and WhatsApp handoff links.
    - **LLM:** used for interpretation and natural-language synthesis.
    - **Location privacy:** device coordinates are only requested on the mechanic-finder flow and are shared with a selected mechanic only after explicit appointment consent. The demo stores appointment state locally in SQLite.
    """)

with tab4:
    st.markdown("### Curated research sources")
    sources = [
        ("NHTSA Tire Safety", "https://www.nhtsa.gov/vehicle-safety/tires", "Tire pressure, tire safety, maintenance guidance."),
        ("Motorcycle Safety Foundation — T-CLOCS", "https://msf-usa.org/documents/library/t-clocs-pre-ride-inspection-checklist/", "Two-wheeler pre-ride checks: tires, brakes, controls, lights, fluids."),
        ("U.S. EPA — OBD resources", "https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100LW9G.TXT", "OBD concepts, emissions monitoring and diagnostic information."),
        ("Hyundai Owner's Manuals", "https://ownersmanual.hyundai.com/", "Manufacturer example for warning lights and safety-specific procedures."),
    ]
    for name, url, desc in sources:
        st.markdown(f'<div class="source"><b>{html.escape(name)}</b><br><span class="small">{html.escape(desc)}</span><br><a href="{html.escape(url)}" target="_blank">{html.escape(url)}</a></div>', unsafe_allow_html=True)

with tab5:
    st.markdown("""
    <div class="card">
      <h3>🛠️ Mechanic Portal — incoming service requests</h3>
      <p class="small">Hackathon demo portal for the shop owner/service desk. Incoming requests include the customer's problem, AI assessment, vehicle, preferred slot and explicitly shared service location.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    pending = get_service_requests(status="PENDING", limit=50)
    all_requests = get_service_requests(limit=50)
    if not all_requests:
        st.info("No service requests are waiting yet. Submit one from the Nearby Mechanics tab to see it here.")
    else:
        st.markdown("### Incoming requests")
        request_options = [r["id"] for r in all_requests]
        selected_request_id = st.selectbox(
            "Select a service request",
            request_options,
            format_func=lambda rid: next((f"{r['id']} • {r['shop_name']} • {r['status']} • {r['customer_name']}" for r in all_requests if r['id'] == rid), rid),
            key="portal_request_id",
        )
        selected_req = get_service_request(selected_request_id)
        if selected_req:
            st.markdown("### Request brief")
            st.markdown(f"""
            <div class="mechanic-card">
              <div class="mechanic-title">{html.escape(str(selected_req['shop_name']))} · {html.escape(str(selected_req['id']))}</div>
              <div class="mechanic-meta">Status: {html.escape(str(selected_req['status']))} • Preferred slot: {html.escape(str(selected_req.get('preferred_slot') or 'Not specified'))}</div>
              <br><b>Customer:</b> {html.escape(str(selected_req['customer_name']))} · {html.escape(str(selected_req['customer_phone']))}<br>
              <b>Vehicle:</b> {html.escape(str(selected_req['vehicle']))}<br>
              <b>Problem:</b> {html.escape(str(selected_req['problem']))}<br>
              <b>Service location:</b> {html.escape(str(selected_req['service_location_label']))}<br>
              <b>AI assessment:</b><br>{html.escape(str(selected_req['diagnosis'])[:2500])}
            </div>
            """, unsafe_allow_html=True)

            if selected_req.get("shop_lat") is not None and selected_req.get("shop_lon") is not None:
                maps_url = f"https://www.google.com/maps/search/?api=1&query={selected_req['shop_lat']},{selected_req['shop_lon']}"
                st.link_button("🗺️ Open selected shop in Maps", maps_url)

            st.markdown("### Owner decision")
            with st.form(f"owner_form_{selected_request_id}"):
                c1, c2 = st.columns(2)
                with c1:
                    mechanic_name = st.text_input("Mechanic / owner name", value=selected_req.get("mechanic_name") or "")
                    mechanic_phone = st.text_input("Mechanic contact number", value=selected_req.get("mechanic_phone") or "")
                with c2:
                    eta_minutes = st.number_input("ETA (minutes)", min_value=0, max_value=1440, value=int(selected_req.get("eta_minutes") or 30), step=5)
                    estimated_fee = st.text_input("Estimated service/deal fee", value=selected_req.get("estimated_fee") or "")
                deal_note = st.text_area("Deal / appointment note", value=selected_req.get("deal_note") or "", placeholder="e.g. Inspection + roadside visit; final parts cost after inspection.")
                owner_message = st.text_area("Message to customer", value=selected_req.get("owner_message") or "", placeholder="e.g. I can reach you in 25 minutes. Please keep the vehicle safely parked.")
                accept = st.form_submit_button("✅ ACCEPT + SET ETA + CONFIRM DEAL", type="primary")
                reject = st.form_submit_button("❌ REJECT REQUEST")
                if accept:
                    if not mechanic_name.strip() or not mechanic_phone.strip():
                        st.error("Enter the mechanic name and contact number before accepting so AutoSage can give the customer a real contact.")
                    else:
                        updated = update_service_request(
                            selected_request_id,
                            status="ACCEPTED",
                            mechanic_name=mechanic_name,
                            mechanic_phone=mechanic_phone,
                            eta_minutes=int(eta_minutes),
                            estimated_fee=estimated_fee,
                            deal_note=deal_note,
                            owner_message=owner_message,
                        )
                        st.success(build_customer_confirmation_message(updated or selected_req))
                        st.rerun()
                elif reject:
                    update_service_request(selected_request_id, status="REJECTED", owner_message=owner_message or "The shop is unavailable for this request.")
                    st.warning("Request rejected. The customer will see the updated status on refresh.")
                    st.rerun()

            if selected_req.get("shop_phone"):
                wa_url = build_whatsapp_link(selected_req["shop_phone"], build_mechanic_notification_message(selected_req))
                if wa_url:
                    st.link_button("📲 Open WhatsApp to the shop", wa_url)
            else:
                st.caption("Selected shop has no public phone listed in OpenStreetMap. The demo portal still receives the request; production can connect this to a verified business messaging API.")

    st.markdown("### Production handoff")
    st.info("This hackathon build uses SQLite for a deterministic demo workflow and user-triggered WhatsApp links where a public shop number exists. It does not silently send messages or invent mechanic contact details. For production, replace SQLite with a hosted database, add authenticated mechanic accounts, and connect a verified dispatch/WhatsApp/SMS provider for automatic notifications.")

st.markdown("<div style='text-align:center;color:#707a97;margin-top:30px;font-size:.8rem'>AutoSage AI • hackathon prototype • diagnosis + voice + Gujarati + mechanic dispatch • user-controlled service handoff</div>", unsafe_allow_html=True)
