"""LangChain-based agent orchestration: plan -> tool -> RAG -> synthesis."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import get_chat_model
from rag_engine import retrieve
from tools import (
    TOOL_REGISTRY,
    TOOLS,
    extract_obd_code,
    safety_check,
    vehicle_system_triage,
)


SYSTEM_PROMPT = """You are AutoSage, a vehicle troubleshooting assistant for two-wheelers and four-wheelers.

You explain likely causes and practical next checks; you never claim a remote diagnosis is certain.

Use the retrieved evidence and tool output. Prefer inspection/testing before part replacement.

For safety-critical symptoms, recommend safe stopping and professional inspection.

Never provide instructions that would require unsafe road testing or dismantling safety-critical systems.
"""


PLANNER_PROMPT = """Choose exactly ONE tool for this vehicle query, or choose none.

Available LangChain tools:

- obd_code_lookup: when a P0xxx/P1xxx-style diagnostic code is present.
- safety_check: when the symptom may create immediate safety risk (brakes, steering, fuel leak, severe overheating, smoke, flashing check-engine plus rough running).
- vehicle_system_triage: when the user describes symptoms but no stronger specialized tool is needed.
- maintenance_check: when the main request is scheduled/preventive maintenance.
- parts_action_plan: when the user explicitly asks what component may need inspection/replacement.

Return JSON only:
{"tool":"tool_name_or_none","reason":"one sentence"}
"""


def _parse_json(text: str) -> dict[str, Any]:
    """Parse planner JSON safely."""

    text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)

        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return {
        "tool": "none",
        "reason": "Planner output could not be parsed.",
    }


def _fallback_tool(vehicle_type: str, query: str) -> str:
    """Choose a deterministic tool if planner output is unavailable."""

    if extract_obd_code(query):
        return "obd_code_lookup"

    safety = safety_check(query)

    if safety["level"] != "ROUTINE":
        return "safety_check"

    return "vehicle_system_triage"


def _run_selected_tool(
    tool_name: str,
    vehicle_type: str,
    issue: str,
) -> dict[str, Any]:
    """Invoke the selected LangChain StructuredTool."""

    tool_obj = TOOL_REGISTRY.get(tool_name)

    if tool_obj is None:
        return {
            "tool": "none",
            "message": "No specialized tool was needed.",
        }

    if tool_name == "obd_code_lookup":
        code = extract_obd_code(issue) or "UNKNOWN"
        return tool_obj.invoke({"code": code})

    if tool_name in {"safety_check", "vehicle_system_triage"}:
        return tool_obj.invoke({"symptoms": issue})

    if tool_name == "maintenance_check":
        return tool_obj.invoke(
            {
                "vehicle_type": vehicle_type,
                "issue": issue,
            }
        )

    if tool_name == "parts_action_plan":
        triage = TOOL_REGISTRY["vehicle_system_triage"].invoke(
            {"symptoms": issue}
        )

        system = (
            triage["systems"][0]
            if triage.get("systems")
            else "general"
        )

        return tool_obj.invoke(
            {
                "system": system,
                "symptoms": issue,
            }
        )

    return {
        "tool": "none",
        "message": "No specialized tool was needed.",
    }


def _planner_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_PROMPT),
            (
                "human",
                "Vehicle type: {vehicle_type}\n"
                "Vehicle: {vehicle}\n"
                "Problem: {query}",
            ),
        ]
    )

    return prompt | get_chat_model(temperature=0.0) | StrOutputParser()


def _answer_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{prompt}"),
        ]
    )

    return prompt | get_chat_model(temperature=0.0) | StrOutputParser()


def detect_language(query: str) -> str:
    """Detect English, Hindi, Hinglish, Gujarati or Roman Gujarati."""

    # Gujarati script
    if re.search(r"[\u0A80-\u0AFF]", query):
        return "gujarati"

    # Hindi / Devanagari
    if re.search(r"[\u0900-\u097F]", query):
        return "hindi"

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        query.lower(),
    )

    tokens = set(text.split())

    # Roman Gujarati
    gujarati_phrases = [
        "nathi",
        "thati",
        "thai",
        "chhe",
        "che",
        "mari",
        "maru",
        "mane",
        "tamari",
        "tamaru",
        "shu",
        "kem",
        "kyare",
        "karvu",
        "karavi",
        "thay",
        "thayu",
        "aave",
        "aavto",
        "aavti",
        "lage",
        "lage chhe",
        "mathi",
        "saru",
        "nathi thati",
        "nathi thato",
    ]

    gujarati_token_markers = {
        "mari",
        "maru",
        "mane",
        "tamari",
        "tamaru",
        "shu",
        "kem",
        "kyare",
        "karvu",
        "karavi",
        "nathi",
        "thati",
        "thato",
        "thai",
        "thay",
        "chhe",
        "che",
        "aave",
        "aavti",
        "aavto",
        "mathi",
        "lage",
        "barabar",
    }

    # Roman Hindi / Hinglish
    hinglish_phrases = [
        "start nahi",
        "nahi ho",
        "problem kya",
        "check karo",
        "brake lagane",
        "band ho",
        "aa raha",
        "aa rahi",
        "rahi hai",
        "raha hai",
        "hai kya",
        "batao",
        "bata do",
    ]

    hinglish_token_markers = {
        "meri",
        "mujhe",
        "mera",
        "gaadi",
        "gadi",
        "scooty",
        "karu",
        "karo",
        "kyu",
        "kyun",
        "awaz",
        "awaaz",
        "chal",
        "chalu",
        "garam",
        "dikkat",
        "kaise",
        "kab",
        "bahut",
        "zyada",
        "kam",
        "mein",
        "pe",
        "wala",
        "nahi",
    }

    gujarati_phrase_score = sum(
        1 for phrase in gujarati_phrases if phrase in text
    )

    gujarati_token_score = len(
        tokens & gujarati_token_markers
    )

    hinglish_phrase_score = sum(
        1 for phrase in hinglish_phrases if phrase in text
    )

    hinglish_token_score = len(
        tokens & hinglish_token_markers
    )

    if gujarati_phrase_score + gujarati_token_score >= 2:
        return "roman_gujarati"

    if hinglish_phrase_score + hinglish_token_score >= 2:
        return "hinglish"

    return "english"


def cleanup_hinglish(text: str) -> str:
    """Clean common Gujarati leakage from Hinglish output."""

    replacements = [
        (r"\btamari\b", "aapki"),
        (r"\btamaru\b", "aapka"),
        (r"\btamare\b", "aapko"),
        (r"\btamne\b", "aapko"),
        (r"\bchhe\b", "hai"),
        (r"\bnathi\b", "nahi"),
        (r"\bkarvu\b", "karna"),
        (r"\bkarvi\b", "karni"),
        (r"\bjoiye\b", "chahiye"),
        (r"\bmate\b", "liye"),
        (r"\bpan\b", "lekin"),
        (r"\bjo\b", "agar"),
        (r"\bshakay\b", "sakta hai"),
        (r"\bthai shake chhe\b", "ho sakta hai"),
        (r"\bpehla\b", "pehle"),
        (r"\bsharu ma\b", "pehle"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return text


def cleanup_roman_gujarati(text: str) -> str:
    """Normalize mixed Hindi phrases into Roman Gujarati."""

    replacements = [
          # Hindi leakage -> natural Roman Gujarati
        # Verb + Hindi "sakta/sakti/sakte" -> Gujarati
        (
            r"\bkar\s+(sakta|sakti|sakte)\s+(hai|hain|chhe|che)\b",
            "kari shake chhe",
        ),
        (
            r"\bho\s+(sakta|sakti|sakte)\s+(hai|hain|chhe|che)\b",
            "hoi shake chhe",
        ),

        # Any remaining Hindi possibility words
        (
            r"\b(sakta|sakti|sakte)\s+(hai|hain|chhe|che)\b",
            "shake chhe",
        ),

        # Common Hindi phrases
        (
            r"\bkarna chahiye\b",
            "karvu joiye",
        ),
        (
            r"\bkarni chahiye\b",
            "karvi joiye",
        ),
        (
            r"\bkarein\b",
            "karo",
        ),
        (
            r"\bcheck karein\b",
            "check karo",
        ),
        (
            r"\bpehle\b",
            "sharu ma",
        ),
        (
            r"\bpahle\b",
            "sharu ma",
        ),
        (
            r"\blekin\b",
            "pan",
        ),
        (
            r"\bzarurat\b",
            "jaruriyat",
        ),
        (
            r"\baapko\b",
            "tamare",
        ),
        (
            r"\baapki\b",
            "tamari",
        ),
        (
            r"\baapka\b",
            "tamaru",
        ),
        (
            r"\bmujhe\b",
            "mane",
        ),
        (
            r"\bnahi\b",
            "nathi",
        ),
        (
            r"\bhai\b",
            "chhe",
        ),

        # More natural Gujarati sentence connection
        (
            r",\s*jo\s+",
            ", je ",
        ),

        # Common wording cleanup
        (
            r"\btamari engine\b",
            "tamaru engine",
        ),
        (
            r"\btamari engine oil\b",
            "tamaru engine oil",
        ),
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    # Remove accidental repeated words.
    text = re.sub(
        r"\b([A-Za-z]+)(\s+\1\b)+",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    # Remove repeated "Sharu ma,"
    text = re.sub(
        r"(Sharu ma,\s*){2,}",
        "Sharu ma, ",
        text,
        flags=re.IGNORECASE,
    )

    return text


def cleanup_markdown_bullets(text: str) -> str:
    """
    Normalize malformed LLM bullet formatting into
    one simple Markdown bullet style: '- text'.
    """

    lines = text.splitlines()
    cleaned_lines: list[str] = []

    previous_bullet = None

    for line in lines:
        stripped = line.strip()

        # Remove completely empty bullet/number lines.
        if stripped in {
            "-",
            "*",
            "•",
            "1.",
            "2.",
            "3.",
            "4.",
            "5.",
            "6.",
            "7.",
            "8.",
            "9.",
        }:
            continue

        # Remove accidental SVG anchor text.
        line = re.sub(
            r"\[svg\]\(http://localhost:\d+/#.*?\)",
            "",
            line,
            flags=re.IGNORECASE,
        )

        # Match bullet/number markers at the beginning.
        bullet_match = re.match(
            r"^\s*(?:(?:[-*•]+\s*)+|(?:\d+\.\s*)+)(.*)$",
            line,
        )

        if bullet_match:
            content = bullet_match.group(1).strip()

            # Ignore empty bullet content.
            if not content:
                continue

            # Remove leftover bullet symbols from content.
            content = re.sub(
                r"^(?:[-*•]\s*)+",
                "",
                content,
            ).strip()

            # Remove leftover standalone numbering.
            if re.fullmatch(
                r"\d+\.",
                content,
            ):
                continue

            normalized = re.sub(
                r"\s+",
                " ",
                content,
            ).strip().lower()

            # Skip exact duplicate bullet immediately following another.
            if normalized == previous_bullet:
                continue

            line = f"- {content}"
            previous_bullet = normalized

        else:
            previous_bullet = None

        cleaned_lines.append(line.rstrip())

    return "\n".join(cleaned_lines)


def analyze(
    vehicle_type: str,
    make: str,
    model: str,
    year: str,
    fuel: str,
    query: str,
    language_mode: str = "Auto",
) -> dict[str, Any]:

    vehicle = f"{year} {make} {model} ({fuel})".strip()

    detected_language = detect_language(query)

    print(
        "DEBUG detected_language:",
        detected_language,
    )

    mode = (
        language_mode
        .lower()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if mode in {
        "hinglish",
        "roman_hindi",
        "romanhindi",
    }:
        response_language = (
            "Hinglish (Roman Hindi + English)"
        )

    elif mode == "hindi":
        response_language = (
            "Hindi (Devanagari, with English "
            "technical terms where useful)"
        )

    elif mode == "gujarati":
        response_language = (
            "Gujarati (Gujarati script, with English "
            "technical terms where useful)"
        )

    elif mode in {
        "roman_gujarati",
        "romangujarati",
    }:
        response_language = (
            "Roman Gujarati (Gujarati language written "
            "using the English/Roman alphabet)"
        )

    elif mode == "english":
        response_language = "English"

    else:
        response_language = {
            "hinglish": (
                "Hinglish (Roman Hindi + English)"
            ),
            "hindi": (
                "Hindi (Devanagari, with English "
                "technical terms where useful)"
            ),
            "gujarati": (
                "Gujarati (Gujarati script, with English "
                "technical terms where useful)"
            ),
            "roman_gujarati": (
                "Roman Gujarati (Gujarati language written "
                "using the English/Roman alphabet)"
            ),
        }.get(
            detected_language,
            "English",
        )

    print(
        "DEBUG response_language:",
        response_language,
    )

    # ---------------------------------------------------------
    # 1. FAST TOOL SELECTION
    # ---------------------------------------------------------

    query_lower = query.lower()

    if extract_obd_code(query):
        tool_name = "obd_code_lookup"
        planner_reason = "Diagnostic code detected."

    elif any(
        keyword in query_lower
        for keyword in [
            "brake",
            "steering",
            "fuel leak",
            "petrol leak",
            "diesel leak",
            "smoke",
            "burning smell",
            "overheating",
            "engine overheating",
            "flashing check engine",
        ]
    ):
        tool_name = "safety_check"
        planner_reason = "Potential safety-critical symptom detected."

    elif any(
        keyword in query_lower
        for keyword in [
            "service",
            "servicing",
            "maintenance",
            "oil change",
            "periodic service",
            "scheduled maintenance",
        ]
    ):
        tool_name = "maintenance_check"
        planner_reason = "Maintenance-related request detected."

    elif any(
        keyword in query_lower
        for keyword in [
            "replace",
            "replacement",
            "which part",
            "what part",
            "component",
            "spare part",
        ]
    ):
        tool_name = "parts_action_plan"
        planner_reason = "Part/service request detected."

    else:
        tool_name = "vehicle_system_triage"
        planner_reason = "General vehicle symptom detected."

    plan = {
        "tool": tool_name,
        "reason": planner_reason,
    }

    # ---------------------------------------------------------
    # 2. TOOL
    # ---------------------------------------------------------

    tool_result = _run_selected_tool(
        tool_name,
        vehicle_type,
        query,
    )

    # ---------------------------------------------------------
    # 3. RAG
    # ---------------------------------------------------------

    rag_query = (
        f"Vehicle type={vehicle_type}; "
        f"vehicle={vehicle}; "
        f"symptom={query}; "
        f"tool result={tool_result}"
    )

    evidence = retrieve(
        rag_query,
        vehicle_type,
        n_results=3,
    )

    evidence_text = "\n\n".join(
        (
            f"SOURCE {i + 1}: "
            f"{d['metadata'].get('title', 'Untitled')} | "
            f"{d['metadata'].get('source_name', 'Unknown')}\n"
            f"{d['text']}"
        )
        for i, d in enumerate(evidence)
    )

    # ---------------------------------------------------------
    # 4. LANGUAGE RULES
    # ---------------------------------------------------------

    response_language_lower = response_language.lower()

    if response_language_lower.startswith(
        "roman gujarati"
    ):
        language_rules = """
IMPORTANT — ROMAN GUJARATI ONLY

The user is communicating in Gujarati written in Roman/English letters.

The COMPLETE ANSWER must be natural Roman Gujarati.

- Use Gujarati grammar and sentence structure.
- Use ONLY English/Roman alphabet.
- Never use Gujarati script.
- Never switch to Hindi, Hinglish, or English.
- Do not translate Gujarati sentences into English.
- Do not add English explanations in brackets or parentheses.
- Automotive technical terms may remain in English.

Use natural Gujarati expressions such as:
chhe, nathi, mate, tamare, tamari, mane, shu, kem,
jo, pan, karvu, karavo, joiye, hoi shake chhe,
thay shake chhe, kari shake chhe, shake chhe,
pehla, pachhi, check karvu.

IMPORTANT:
- Always use "kari shake chhe", never "kar sakta chhe".
- Always use "hoi shake chhe", never "ho sakta chhe".
- Never use "sakta", "sakti", or "sakte" in the final answer.

Do not use Hindi expressions such as:
hai, hain, ho sakta hai, ho sakte hain, lekin,
mujhe, aapko, pehle, zarurat, karna chahiye.

FORMATTING:
- Use one simple bullet per item.
- Every bullet starts with "- ".
- Never use "*" or numbered bullets.
- Never output a standalone "-".
- Never add an English translation after a Gujarati bullet.
"""

    elif response_language_lower.startswith(
        "hinglish"
    ):
        language_rules = """
IMPORTANT — HINGLISH ONLY

The user is communicating in Hinglish.

The COMPLETE ANSWER must be natural Hinglish using Roman script.

- Use Hindi/Roman Hindi grammar naturally.
- English automotive and technical terms may remain in English.
- Do not switch to Gujarati or Roman Gujarati.
- Do not write the whole answer as formal English.

Use natural expressions such as:
hai, hain, mein, ke, ka, ki, ko, se, aur, lekin,
agar, iska, kya, ho sakta hai, karna, karni,
check karna, dekhna, batana, chahiye, pehle,
baad mein, problem, wajah, zaroori.

Do not use Gujarati expressions such as:
chhe, nathi, tamari, tamare, mate, karvu, joiye, shakay.

FORMATTING:
- Use one simple bullet per item.
- Every bullet starts with "- ".
- Never use "*" or numbered bullets for regular lists.
"""

    elif response_language_lower.startswith(
        "hindi"
    ):
        language_rules = """
IMPORTANT — HINDI ONLY

- Write the COMPLETE answer in Hindi using Devanagari script.
- Technical automotive terms may remain in English.
- Do not switch to Gujarati, Roman Gujarati, or Hinglish.

FORMATTING:
- Use one simple bullet per item.
- Every bullet starts with "- ".
"""

    elif response_language_lower.startswith(
        "gujarati"
    ):
        language_rules = """
IMPORTANT — GUJARATI ONLY

- Write the COMPLETE answer in Gujarati script.
- Technical automotive terms may remain in English.
- Do not switch to Hindi, Roman Gujarati, or Hinglish.

FORMATTING:
- Use one simple bullet per item.
- Every bullet starts with "- ".
"""

    else:
        language_rules = """
IMPORTANT — ENGLISH ONLY

- Write the COMPLETE answer in clear, natural English.
- Use standard English grammar and vocabulary.
- Do not use Hindi, Hinglish, Gujarati, or Roman Gujarati.
- Technical automotive terms may remain in English.

FORMATTING:
- Use one simple bullet per item.
- Every bullet starts with "- ".
"""

    # ---------------------------------------------------------
    # 5. FINAL SYNTHESIS PROMPT
    # ---------------------------------------------------------

    final_prompt = f"""{SYSTEM_PROMPT}

Vehicle: {vehicle}

Vehicle type: {vehicle_type}

User problem:
{query}

Preferred response language:
{response_language}

{language_rules}

LANGCHAIN TOOL USED:
{tool_name}

TOOL RESULT:
{json.dumps(tool_result, indent=2)}

RETRIEVED EVIDENCE FROM LANGCHAIN + CHROMA:
{evidence_text}

Create a clear answer in Markdown using these headings exactly:

### Assessment

### Likely causes

### What to check now

### What may need service or replacement

### Attention level

### When to stop driving

### Sources

DIAGNOSTIC QUALITY RULES:
- Make the likely causes specific to the user's exact symptoms, vehicle type, and reported conditions.
- Do not use generic statements such as "multiple causes can produce this symptom" as the main answer.
- For each likely cause, briefly explain why it matches the reported symptom.
- Give practical, safe checks that help distinguish between the likely causes.
- If a warning light such as the check-engine light is reported, recommend reading the OBD-II trouble code when appropriate.
- For unusual engine noise, identify the type of noise as an important diagnostic clue and ask the user to note whether it is clicking, grinding, squealing, knocking, rattling, or another sound.
- Do not invent a specific failed component or claim a confirmed diagnosis without evidence.
- Prioritize the most relevant 2-4 possibilities rather than listing unrelated possibilities.
- Connect each "What to check now" item to a likely cause whenever possible.
- Do not suggest suspension or steering faults for an engine-startup noise unless the user reports that the noise occurs during steering, braking, bumps, or vehicle movement.
- For a startup noise with a check-engine light, prioritize starting/charging, ignition/fueling, engine/accessory, and exhaust-related possibilities when supported by the evidence.

SAFETY OUTPUT RULES:

- The "Attention level" must be EXACTLY one of: ROUTINE, HIGH, or CRITICAL.
- Do not add any explanation, bullet, or sentence on the "Attention level" line.
- Never upgrade ROUTINE to HIGH or CRITICAL unless the tool result explicitly indicates HIGH or CRITICAL.
- For noise + check-engine-light symptoms, do not automatically tell the user to stop driving.
- Recommend stopping driving only when there is a specific safety reason such as severe overheating, smoke, fire/fuel leak, braking or steering problems, loss of control, flashing check-engine light with severe rough running, or another clearly dangerous symptom.
- If the check-engine light is steady and the vehicle is otherwise operating normally, explain that the vehicle should be inspected and the OBD-II code should be read, rather than automatically saying to stop driving.
- Under "When to stop driving", list only concrete conditions that justify stopping.
- Do not write phrases such as "Safety critical:**" inside the answer.

FORMATTING RULES:
- Assessment should be 1-3 short sentences.
- Under Likely causes, use separate bullet points.
- Under What to check now, use separate bullet points.
- Under What may need service or replacement, use separate bullet points.
- Under When to stop driving, use separate bullet points when there are multiple conditions.
- Every bullet must use ONLY the "- " Markdown style.
- Never use "*" bullets.
- Never use "1.", "2.", "3." bullets unless a truly ordered sequence is required.
- Never output a standalone "-" bullet.
- Never output duplicate bullets.
- Keep each bullet focused on one point.

SAFETY AND CONTENT RULES:
- Distinguish likely causes from confirmed faults.
- Do not invent exact repair costs or mileage intervals.
- For a diagnostic code, explain that the code narrows a system and does not prove a specific part is bad.
- For critical/high safety findings from the tool, make the safety action prominent.
- Under Sources, list 2-4 retrieved sources as markdown links.
- Keep the COMPLETE answer in the requested language.
"""

    # ---------------------------------------------------------
    # 6. ANSWER
    # ---------------------------------------------------------

    try:
        answer = _answer_chain().invoke(
            {"prompt": final_prompt}
        )

    except Exception as exc:
        print("DEBUG answer_chain error:", repr(exc))
        answer = _fallback_answer(
            vehicle,
            query,
            tool_result,
            evidence,
            response_language,
        )

    # Language cleanup
    if response_language_lower.startswith(
        "roman gujarati"
    ):
        answer = cleanup_roman_gujarati(answer)

    elif response_language_lower.startswith(
        "hinglish"
    ):
        answer = cleanup_hinglish(answer)

    # Markdown cleanup
    answer = answer.replace(r"\*\*", "**")
    answer = answer.replace(r"\*", "*")
    answer = answer.replace(r"\_", "_")
    answer = answer.replace(r"\#", "#")

    # Attention level should never be shown as a bullet.
    answer = re.sub(
        r"(### Attention level\s*)-\s*(ROUTINE|HIGH|CRITICAL)\b",
        r"\1\2",
        answer,
        flags=re.IGNORECASE,
    )

    answer = cleanup_markdown_bullets(answer)

    mechanic_recommended = (
        str(tool_result.get("level", "")).upper()
        in {"HIGH", "CRITICAL"}
    )

    return {
        "answer": answer,
        "detected_language": detected_language,
        "response_language": response_language,
        "vehicle": vehicle,
        "tool": tool_name,
        "tool_result": tool_result,
        "planner_reason": plan.get("reason", ""),
        "evidence": evidence,
        "mechanic_recommended": mechanic_recommended,
        "framework": "LangChain",
    }


def _fallback_answer(
    vehicle: str,
    query: str,
    tool_result: dict[str, Any],
    evidence: list[dict[str, Any]],
    response_language: str = "English",
) -> str:

    response_language_lower = response_language.lower()

    source_links = [
        (
            f"- [{d['metadata'].get('title', 'Source')}]"
            f"({d['metadata'].get('source_url', '#')})"
        )
        for d in evidence[:4]
    ]

    # ---------------------------------------------------------
    # HINGLISH FALLBACK
    # ---------------------------------------------------------

    if response_language_lower.startswith("hinglish"):

        lines = [
            "### Assessment",
            (
                f"**{vehicle}** ke liye aapne jo symptom "
                f"bataya hai: _{query}_"
            ),
        ]

        if tool_result.get("meaning"):

            lines += [
                "\n### Likely causes",
                (
                    f"- **{tool_result.get('meaning')}** ka matlab "
                    "generic diagnostic-code level par ye hai."
                ),
                (
                    "- Sirf code ke basis par kisi part ko failed "
                    "confirm nahi karna chahiye."
                ),
                "\n### What to check now",
            ]

            for item in tool_result.get("checks", []):
                lines.append(f"- {item}")

            lines += [
                "\n### What may need service or replacement",
                (
                    f"- {tool_result.get('action', 'Pehle inspect aur '
                    'test karein; replacement fault confirm hone '
                    'ke baad hi karein.')}"
                ),
            ]

        else:

            lines += [
                "\n### Likely causes",
                (
                    "- Is symptom ke multiple possible causes ho "
                    "sakte hain; ye troubleshooting guidance hai, "
                    "confirmed diagnosis nahi."
                ),
                "\n### What to check now",
                (
                    "- Symptom exactly kab hota hai note karein."
                ),
                (
                    "- Warning lights, leaks aur visible damage ko "
                    "safely observe karein."
                ),
                (
                    "- Hot ya moving parts ko touch na karein."
                ),
                "\n### What may need service or replacement",
                (
                    "- Part replace karne se pehle affected system "
                    "ko diagnose karein."
                ),
            ]

        lines += [
            "\n### Attention level",
            str(tool_result.get("level", "ROUTINE")),
            "\n### When to stop driving",
        ]

        action = tool_result.get(
            "action",
            (
                "Agar control, braking ya fire/fuel-leak risk "
                "ho, vehicle ko safely stop karke professional "
                "help lein."
            ),
        )

        if isinstance(action, str):
            lines.append(action)
        else:
            lines.append(str(action))

        lines += [
            "\n### Sources",
            *source_links,
        ]

        return "\n".join(lines)

    # ---------------------------------------------------------
    # ROMAN GUJARATI FALLBACK
    # ---------------------------------------------------------

    if response_language_lower.startswith(
        "roman gujarati"
    ):

        lines = [
            "### Assessment",
            (
                f"**{vehicle}** mate tame aa symptom "
                f"janavyo chhe: _{query}_"
            ),
            "\n### Likely causes",
            (
                "- Aa symptom na ghana sambhavit karano hoi shake chhe."
            ),
            (
                "- Aa troubleshooting guidance chhe, "
                "confirmed diagnosis nathi."
            ),
            "\n### What to check now",
            (
                "- Warning lights, leaks ane visible damage "
                "safely observe karo."
            ),
            (
                "- Hot athva moving parts ne touch na karo."
            ),
            "\n### What may need service or replacement",
            (
                "- Koi pan part replace karta pehla affected "
                "system nu proper inspection ane testing karavavu joiye."
            ),
            "\n### Attention level",
            str(tool_result.get("level", "ROUTINE")),
            "\n### When to stop driving",
        ]

        action = tool_result.get(
            "action",
            (
                "Jo braking, steering, fire/fuel-leak athva "
                "severe overheating nu risk hoy, to vehicle "
                "safely rokine professional help lo."
            ),
        )

        if isinstance(action, str):
            lines.append(action)
        else:
            lines.append(str(action))

        lines += [
            "\n### Sources",
            *source_links,
        ]

        return "\n".join(lines)

    # ---------------------------------------------------------
    # ENGLISH FALLBACK
    # ---------------------------------------------------------

    # Keep cloud fallback useful even if the hosted LLM is temporarily
    # unavailable. For the common startup-noise + check-engine case,
    # provide focused, evidence-safe guidance instead of a generic sentence.
    query_lower = query.lower()

    if (
        ("check engine" in query_lower or "check-engine" in query_lower)
        and any(word in query_lower for word in ["noise", "sound", "rattle", "click", "grinding", "squeal"])
        and any(word in query_lower for word in ["start", "starting", "startup", "crank"])
    ):
        return (
            f"### Assessment\n"
            f"**{vehicle}** has a startup noise together with a check-engine light. "
            "The exact noise type and the stored OBD-II code are the most useful clues for narrowing the cause.\n\n"
            "### Likely causes\n"
            "- **Starting/charging issue:** A weak battery, poor terminal connection, or starter-system fault can cause clicking, slow cranking, or unusual starting noise.\n"
            "- **Ignition or fueling issue:** An ignition or fuel-delivery problem can trigger the check-engine light and may cause rough or difficult starting.\n"
            "- **Engine accessory or belt-related noise:** A worn belt, tensioner, pulley, or accessory can create clicking, squealing, or rattling at startup.\n"
            "- **Exhaust/emissions issue:** The check-engine light may come from an emissions-related fault; the OBD-II code is needed before identifying a specific component.\n\n"
            "### What to check now\n"
            "- **Identify the noise:** Note whether it is clicking, grinding, squealing, knocking, rattling, or another sound, and whether it stops after the engine starts.\n"
            "- **Read the OBD-II code:** Scan the check-engine light and record the exact code before replacing parts.\n"
            "- **Check the battery and terminals:** Look for loose/corroded connections and have battery/charging condition tested.\n"
            "- **Observe starting behavior:** Note slow cranking, misfiring, shaking, loss of power, or whether the noise continues after startup.\n\n"
            "### What may need service or replacement\n"
            "- The battery, starter system, ignition/fueling components, or an engine accessory may need service depending on the inspection and OBD-II findings.\n"
            "- Do not replace the catalytic converter, oxygen sensor, or another specific component based only on the check-engine light.\n\n"
            "### Attention level\n"
            f"{tool_result.get('level', 'ROUTINE')}\n\n"
            "### When to stop driving\n"
            "- Stop driving and arrange professional help if the check-engine light is flashing, the engine is severely misfiring, there is smoke/fire/fuel leakage, severe overheating, or loss of vehicle control.\n"
            "- If the light is steady and the car otherwise runs normally, arrange inspection and read the OBD-II code rather than automatically stopping the vehicle.\n\n"
            "### Sources\n"
            + "\n".join(source_links)
        )

    return (
        f"### Assessment\n"
        f"**{vehicle}** — reported symptom: _{query}_\n\n"
        "### Likely causes\n"
        "- The symptom can have multiple possible causes; use the retrieved evidence and tool result as troubleshooting guidance, not a confirmed diagnosis.\n\n"
        "### What to check now\n"
        "- Note exactly when the symptom occurs and what conditions make it better or worse.\n"
        "- Check visible warning lights, leaks, and obvious damage without touching hot or moving parts.\n\n"
        "### What may need service or replacement\n"
        "- Diagnose the affected system before replacing components.\n\n"
        "### Attention level\n"
        f"{tool_result.get('level', 'ROUTINE')}\n\n"
        "### When to stop driving\n"
        f"{tool_result.get('action', 'Stop driving and seek professional help if vehicle control, braking, overheating, fuel leakage, or smoke becomes unsafe.')}\n\n"
        "### Sources\n"
        + "\n".join(source_links)
    )