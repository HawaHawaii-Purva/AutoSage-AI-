"""Deterministic domain tools used by the AutoSage agent.

The LLM explains and orchestrates. High-risk gates and structured lookups remain
plain Python so they are testable and predictable.
"""
from __future__ import annotations

import re
from typing import Any

from langchain.tools import tool
from langchain_core.tools import StructuredTool

OBD_CODES: dict[str, dict[str, Any]] = {
    "P0420": {
        "meaning": "Catalyst system efficiency below threshold (Bank 1)",
        "system": "emissions / catalytic-converter monitoring",
        "checks": [
            "Check for other stored DTCs and freeze-frame information.",
            "Inspect for exhaust leaks or damaged exhaust components.",
            "Have upstream/downstream oxygen-sensor data evaluated with a scan tool.",
            "Confirm engine operation is normal before blaming the catalytic converter."
        ],
        "action": "Do not replace the catalytic converter solely from P0420; diagnose the cause first.",
    },
    "P0300": {
        "meaning": "Random/multiple-cylinder misfire detected",
        "system": "engine combustion / ignition / fuel",
        "checks": [
            "Check whether the engine is running roughly or losing power.",
            "Look for additional misfire-related codes and freeze-frame data.",
            "Have ignition, fuel delivery, and intake-related causes checked."
        ],
        "action": "If the check-engine light is flashing or the engine is severely misfiring, prioritize service and avoid continued hard driving.",
    },
    "P0171": {
        "meaning": "System too lean (Bank 1)",
        "system": "air/fuel control",
        "checks": [
            "Inspect for intake/vacuum leaks.",
            "Check fuel delivery and sensor data.",
            "Look for additional codes that narrow the cause."
        ],
        "action": "Diagnose the air/fuel system before replacing sensors or injectors."
    },
    "P0128": {
        "meaning": "Coolant thermostat temperature below regulating range",
        "system": "engine cooling / thermostat monitoring",
        "checks": [
            "Check whether engine temperature reaches the expected operating range.",
            "Review coolant level and condition when the engine is safely cool.",
            "Have thermostat/coolant-temperature sensor operation checked."
        ],
        "action": "Do not open a hot cooling system; follow the vehicle manual's cooling-system procedure."
    },
    "P0016": {
        "meaning": "Crankshaft/camshaft position correlation fault",
        "system": "engine timing / position sensing",
        "checks": [
            "Check for additional engine timing or sensor codes.",
            "Assess starting, idle, power, and unusual engine-noise symptoms.",
            "Have timing and sensor signals professionally diagnosed."
        ],
        "action": "Avoid guessing at a sensor replacement; timing-related faults can have multiple causes."
    },
}

CRITICAL_TERMS = [
    "brake failure", "brakes failed", "no brakes", "brake pedal goes to floor",
    "steering stuck", "cannot steer", "steering failure", "fuel leak", "petrol leak",
    "diesel leak", "burning electrical smell", "electrical burning", "heavy smoke",
    "engine fire", "overheating badly", "temperature red", "smoke from engine",
]

HIGH_TERMS = [
    "brake grinding", "braking is weak", "reduced braking", "brake warning",
    "oil pressure warning", "engine overheating", "overheated", "steering pulls",
    "wheel wobble", "flashing check engine", "check engine flashing",
]


def extract_obd_code(text: str) -> str | None:
    match = re.search(r"\b(P\d{4})\b", text.upper())
    return match.group(1) if match else None


def obd_code_lookup(code: str) -> dict[str, Any]:
    code = code.upper().strip()
    if code in OBD_CODES:
        result = {"tool": "obd_code_lookup", "code": code, **OBD_CODES[code]}
        result["note"] = "Generic OBD-II meaning; exact vehicle-specific interpretation and repair procedure may differ."
        return result
    return {
        "tool": "obd_code_lookup",
        "code": code,
        "meaning": "Code not present in the MVP lookup table.",
        "action": "Use the vehicle's manufacturer service information or a diagnostic scanner/database for the exact code.",
        "note": "Do not infer a replacement part from the code alone.",
    }


def safety_check(symptoms: str) -> dict[str, Any]:
    text = symptoms.lower()
    critical = [term for term in CRITICAL_TERMS if term in text]
    high = [term for term in HIGH_TERMS if term in text]
    if critical:
        return {
            "tool": "safety_check",
            "level": "CRITICAL",
            "color": "red",
            "matched": critical,
            "action": "Stop in a safe location as soon as safely possible and arrange professional/roadside assistance. Do not continue driving if vehicle control or braking is compromised, or if there is a fire/fuel-leak risk.",
        }
    if high:
        return {
            "tool": "safety_check",
            "level": "HIGH",
            "color": "orange",
            "matched": high,
            "action": "Limit driving and arrange inspection soon. Escalate immediately if braking, steering, smoke, temperature, or vehicle behavior worsens.",
        }
    return {
        "tool": "safety_check",
        "level": "ROUTINE",
        "color": "green",
        "matched": [],
        "action": "No critical safety phrase was detected. Normal caution still applies and symptoms should be verified by inspection when appropriate.",
    }


def vehicle_system_triage(symptoms: str) -> dict[str, Any]:
    text = symptoms.lower()
    categories: list[str] = []
    mapping = {
        "starting/charging": ["won't start", "doesn't start", "not starting", "clicking", "crank", "battery", "weak horn", "dim light"],
        "brakes": ["brake", "grinding", "squeal", "stopping"],
        "cooling": ["overheat", "overheating", "temperature", "coolant", "hot"],
        "engine/combustion": ["jerk", "misfire", "rough idle", "stall", "power loss", "check engine"],
        "tires/wheels": ["tyre", "tire", "puncture", "wobble", "vibration", "pressure"],
        "steering/suspension": ["steering", "pull", "clunk", "knock", "bump", "suspension"],
        "emissions/obd": ["obd", "dtc", "code", "check engine", "mil"],
    }
    for category, terms in mapping.items():
        if any(term in text for term in terms):
            categories.append(category)
    return {"tool": "vehicle_system_triage", "systems": categories or ["general inspection"], "note": "This is symptom triage, not a confirmed diagnosis."}


def maintenance_check(vehicle_type: str, issue: str) -> dict[str, Any]:
    common = [
        "Check the owner's manual for vehicle-specific maintenance intervals.",
        "Inspect tires and visible damage/leaks.",
        "Check warning lights and stored diagnostic codes when applicable.",
    ]
    if vehicle_type == "two_wheeler":
        common += [
            "Before riding, check tire condition/pressure, both brakes, lights, controls, and visible fluid/fuel leaks.",
        ]
    else:
        common += [
            "For a passenger car, inspect tire pressure, brakes, fluid/leak symptoms, and dashboard warnings according to the manual.",
        ]
    return {"tool": "maintenance_check", "checks": common, "issue": issue, "note": "No universal mileage interval is assumed because service schedules vary by make/model/year."}


def parts_action_plan(system: str, symptoms: str) -> dict[str, Any]:
    system = system.lower()
    if "brake" in system:
        action = "Inspect braking system condition first; replacement is appropriate only when pads/rotors or another component are confirmed worn/damaged according to the vehicle procedure."
    elif "battery" in system or "starting" in system:
        action = "Test battery state/charging and connections before buying a replacement."
    elif "tire" in system or "wheel" in system:
        action = "Verify pressure and inspect tread, sidewall, wheel, and visible damage; replace or repair based on inspection/manufacturer criteria."
    else:
        action = "Inspect and test the suspected system before replacing a component. Use vehicle-specific service information for the exact part and procedure."
    return {"tool": "parts_action_plan", "system": system, "recommended_action": action, "note": "Component replacement should follow a confirmed fault, not symptom guessing."}


@tool("obd_code_lookup", description="Look up the generic meaning and checks for a P0xxx/P1xxx diagnostic code.")
def obd_code_lookup_langchain(code: str) -> dict[str, Any]:
    return obd_code_lookup(code)


@tool("safety_check", description="Run the deterministic safety gate for vehicle symptoms.")
def safety_check_langchain(symptoms: str) -> dict[str, Any]:
    return safety_check(symptoms)


@tool("vehicle_system_triage", description="Map symptoms to the vehicle systems that should be inspected first.")
def vehicle_system_triage_langchain(symptoms: str) -> dict[str, Any]:
    return vehicle_system_triage(symptoms)


@tool("maintenance_check", description="Generate preventive maintenance checks for the selected vehicle type.")
def maintenance_check_langchain(vehicle_type: str, issue: str) -> dict[str, Any]:
    return maintenance_check(vehicle_type, issue)


@tool("parts_action_plan", description="Suggest what system to inspect before replacing a component.")
def parts_action_plan_langchain(system: str, symptoms: str) -> dict[str, Any]:
    return parts_action_plan(system, symptoms)


LANGCHAIN_TOOLS: list[StructuredTool] = [
    obd_code_lookup_langchain,
    safety_check_langchain,
    vehicle_system_triage_langchain,
    maintenance_check_langchain,
    parts_action_plan_langchain,
]
TOOL_REGISTRY = {item.name: item for item in LANGCHAIN_TOOLS}
TOOLS = set(TOOL_REGISTRY)


# ---------------- Nearby mechanic finder ----------------
# Uses end-user-triggered OpenStreetMap Nominatim geocoding + Overpass POI search.
# Configure endpoints through environment variables so the provider can be swapped
# if required. No location is persisted by AutoSage itself.
import math
import os
import time
import requests

DEFAULT_OSM_USER_AGENT = "AutoSageAI/1.0-hackathon"
DEFAULT_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geocode_location(query: str) -> dict[str, Any]:
    """Resolve a user-entered city/area/pincode to a map coordinate."""
    query = query.strip()
    if not query:
        return {"ok": False, "error": "Enter a city, area or pincode."}
    headers = {"User-Agent": os.getenv("OSM_USER_AGENT", DEFAULT_OSM_USER_AGENT)}
    url = os.getenv("OSM_NOMINATIM_URL", DEFAULT_NOMINATIM_URL)
    params = {"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "in"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return {"ok": False, "error": f"Could not find '{query}'. Try a larger area or city name."}
        item = data[0]
        return {
            "ok": True,
            "latitude": float(item["lat"]),
            "longitude": float(item["lon"]),
            "display_name": item.get("display_name", query),
            "provider": "OpenStreetMap Nominatim",
        }
    except Exception as exc:
        return {"ok": False, "error": f"Location lookup failed: {type(exc).__name__}. You can try another area/city."}


def find_nearby_mechanics(latitude: float, longitude: float, vehicle_type: str = "four_wheeler", radius_m: int = 5000) -> dict[str, Any]:
    """Find nearby vehicle repair POIs from OpenStreetMap via Overpass."""
    radius_m = max(1000, min(int(radius_m), 10000))
    tags = '("shop"="car_repair";"shop"="motorcycle";"shop"="motorcycle_repair";"amenity"="car_repair";"craft"="vehicle_repair")'
    query = f"""[out:json][timeout:25];\n(\n  nwr(around:{radius_m},{latitude},{longitude})[shop=car_repair];\n  nwr(around:{radius_m},{latitude},{longitude})[shop=motorcycle];\n  nwr(around:{radius_m},{latitude},{longitude})[shop=motorcycle_repair];\n  nwr(around:{radius_m},{latitude},{longitude})[amenity=car_repair];\n  nwr(around:{radius_m},{latitude},{longitude})[craft=vehicle_repair];\n);\nout center tags;"""
    headers = {"User-Agent": os.getenv("OSM_USER_AGENT", DEFAULT_OSM_USER_AGENT)}
    overpass_url = os.getenv("OVERPASS_URL", DEFAULT_OVERPASS_URL)
    try:
        resp = requests.post(overpass_url, data=query, headers=headers, timeout=35)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return {
            "tool": "find_nearby_mechanics",
            "ok": False,
            "error": f"Nearby search is temporarily unavailable ({type(exc).__name__}).",
            "mechanics": [],
        }

    target = "motorcycle" if vehicle_type == "two_wheeler" else "car"
    rows = []
    seen = set()
    for el in payload.get("elements", []):
        tags_obj = el.get("tags", {}) or {}
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        lat, lon = float(lat), float(lon)
        name = tags_obj.get("name") or tags_obj.get("brand") or "Nearby vehicle repair shop"
        key = (name.lower(), round(lat, 5), round(lon, 5))
        if key in seen:
            continue
        seen.add(key)
        shop_tag = tags_obj.get("shop") or tags_obj.get("amenity") or tags_obj.get("craft") or "vehicle repair"
        specialization = "Motorcycle repair" if shop_tag in {"motorcycle", "motorcycle_repair"} else "Car / vehicle repair"
        address_parts = [
            tags_obj.get("addr:housenumber"), tags_obj.get("addr:street"),
            tags_obj.get("addr:suburb"), tags_obj.get("addr:city"), tags_obj.get("addr:postcode")
        ]
        address = ", ".join(x for x in address_parts if x)
        distance = _haversine_km(latitude, longitude, lat, lon)
        compatibility = "Good match" if ((target == "motorcycle" and "Motorcycle" in specialization) or (target == "car" and "Car" in specialization)) else "General vehicle repair"
        rows.append({
            "name": name,
            "specialization": specialization,
            "compatibility": compatibility,
            "address": address or "Address not listed on OpenStreetMap",
            "latitude": lat,
            "longitude": lon,
            "distance_km": round(distance, 2),
            "phone": tags_obj.get("phone") or tags_obj.get("contact:phone") or "",
            "website": tags_obj.get("website") or "",
            "osm_type": el.get("type"),
            "osm_id": el.get("id"),
        })

    rows.sort(key=lambda x: (0 if x["compatibility"] == "Good match" else 1, x["distance_km"]))
    return {
        "tool": "find_nearby_mechanics",
        "ok": True,
        "mechanics": rows[:12],
        "radius_km": round(radius_m / 1000, 1),
        "provider": "OpenStreetMap Overpass",
        "attribution": "© OpenStreetMap contributors",
    }


@tool("geocode_location", description="Resolve a user-entered area, city or pincode into coordinates for the mechanic finder.")
def geocode_location_langchain(query: str) -> dict[str, Any]:
    return geocode_location(query)


@tool("find_nearby_mechanics", description="Find nearby vehicle repair shops from OpenStreetMap using user-provided coordinates.")
def find_nearby_mechanics_langchain(latitude: float, longitude: float, vehicle_type: str = "four_wheeler", radius_m: int = 5000) -> dict[str, Any]:
    return find_nearby_mechanics(latitude, longitude, vehicle_type, radius_m)


LANGCHAIN_LOCATION_TOOLS: list[StructuredTool] = [
    geocode_location_langchain,
    find_nearby_mechanics_langchain,
]
