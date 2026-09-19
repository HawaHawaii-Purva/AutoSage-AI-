"""Service-dispatch workflow for AutoSage's hackathon demo.

This module provides a small SQLite-backed appointment state machine:
user selects a nearby shop -> creates a service request -> mechanic owner
reviews it -> owner accepts/rejects and sets ETA/deal -> user sees confirmation.

For production, replace the local SQLite store with an authenticated hosted
backend and a verified business messaging/dispatch provider.
"""
from __future__ import annotations

import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

DB_PATH = Path(os.getenv("SERVICE_DB_PATH", "autosage_service.db"))


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS service_requests (
                id TEXT PRIMARY KEY,
                user_session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                customer_phone TEXT NOT NULL,
                service_location_label TEXT NOT NULL,
                service_lat REAL,
                service_lon REAL,
                vehicle TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                problem TEXT NOT NULL,
                diagnosis TEXT NOT NULL,
                diagnostic_tool TEXT,
                shop_name TEXT NOT NULL,
                shop_address TEXT,
                shop_lat REAL,
                shop_lon REAL,
                shop_phone TEXT,
                preferred_slot TEXT,
                status TEXT NOT NULL DEFAULT 'PENDING',
                mechanic_name TEXT,
                mechanic_phone TEXT,
                eta_minutes INTEGER,
                estimated_fee TEXT,
                deal_note TEXT,
                owner_message TEXT
            )
            """
        )
        conn.commit()


init_db()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _clean_phone(phone: str) -> str:
    return re.sub(r"[^0-9+]", "", (phone or "").strip())


def create_service_request(
    *,
    user_session_id: str,
    customer_name: str,
    customer_phone: str,
    service_location_label: str,
    service_lat: float | None,
    service_lon: float | None,
    vehicle: str,
    vehicle_type: str,
    problem: str,
    diagnosis: str,
    diagnostic_tool: str,
    shop: dict[str, Any],
    preferred_slot: str,
) -> dict[str, Any]:
    """Create a pending mechanic appointment request."""
    init_db()
    request_id = "AS-" + uuid.uuid4().hex[:8].upper()
    now = datetime.now(timezone.utc).isoformat()
    payload = (
        request_id,
        user_session_id,
        now,
        customer_name.strip(),
        _clean_phone(customer_phone),
        service_location_label.strip(),
        service_lat,
        service_lon,
        vehicle.strip(),
        vehicle_type,
        problem.strip(),
        diagnosis.strip(),
        diagnostic_tool.strip(),
        str(shop.get("name") or "Selected repair shop").strip(),
        str(shop.get("address") or "").strip(),
        shop.get("latitude"),
        shop.get("longitude"),
        _clean_phone(str(shop.get("phone") or "")),
        preferred_slot.strip(),
        "PENDING",
        None,
        None,
        None,
        None,
        None,
        None,
    )
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO service_requests (
                id, user_session_id, created_at, customer_name, customer_phone,
                service_location_label, service_lat, service_lon, vehicle,
                vehicle_type, problem, diagnosis, diagnostic_tool, shop_name,
                shop_address, shop_lat, shop_lon, shop_phone, preferred_slot,
                status, mechanic_name, mechanic_phone, eta_minutes,
                estimated_fee, deal_note, owner_message
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            payload,
        )
        conn.commit()
    return get_service_request(request_id) or {"id": request_id, "status": "PENDING"}


def get_service_request(request_id: str) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM service_requests WHERE id = ?", (request_id,)).fetchone()
    return _row_to_dict(row)


def get_service_requests(
    *,
    user_session_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List recent service requests, optionally scoped to a user or status."""
    init_db()
    clauses: list[str] = []
    params: list[Any] = []
    if user_session_id:
        clauses.append("user_session_id = ?")
        params.append(user_session_id)
    if status:
        clauses.append("status = ?")
        params.append(status.upper())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), 200)))
    query = f"SELECT * FROM service_requests {where} ORDER BY created_at DESC LIMIT ?"
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def update_service_request(
    request_id: str,
    *,
    status: str,
    mechanic_name: str | None = None,
    mechanic_phone: str | None = None,
    eta_minutes: int | None = None,
    estimated_fee: str | None = None,
    deal_note: str | None = None,
    owner_message: str | None = None,
) -> dict[str, Any] | None:
    """Update appointment status and mechanic-confirmation details."""
    normalized = status.upper().strip()
    allowed = {"PENDING", "ACCEPTED", "REJECTED", "COMPLETED"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported service request status: {status}")
    eta = int(eta_minutes) if eta_minutes is not None else None
    if eta is not None:
        eta = max(0, min(eta, 1440))
    with _connect() as conn:
        conn.execute(
            """
            UPDATE service_requests
            SET status = ?, mechanic_name = ?, mechanic_phone = ?, eta_minutes = ?,
                estimated_fee = ?, deal_note = ?, owner_message = ?
            WHERE id = ?
            """,
            (
                normalized,
                (mechanic_name or "").strip() or None,
                _clean_phone(mechanic_phone or "") or None,
                eta,
                (estimated_fee or "").strip() or None,
                (deal_note or "").strip() or None,
                (owner_message or "").strip() or None,
                request_id,
            ),
        )
        conn.commit()
    return get_service_request(request_id)


def build_whatsapp_link(phone: str, message: str) -> str:
    """Create a user-triggered WhatsApp chat link with a prefilled message."""
    cleaned = _clean_phone(phone).replace("+", "")
    if cleaned.startswith("0") and len(cleaned) == 11:
        cleaned = "91" + cleaned[1:]
    elif len(cleaned) == 10 and cleaned.isdigit():
        cleaned = "91" + cleaned
    return f"https://wa.me/{cleaned}?text={quote(message)}" if cleaned else ""


def build_mechanic_notification_message(request: dict[str, Any]) -> str:
    """Prepare the structured problem summary for a mechanic owner."""
    return (
        f"AutoSage service request {request.get('id', '')}\n"
        f"Customer: {request.get('customer_name', '')}\n"
        f"Contact: {request.get('customer_phone', '')}\n"
        f"Vehicle: {request.get('vehicle', '')}\n"
        f"Problem reported: {request.get('problem', '')}\n"
        f"AI assessment: {request.get('diagnosis', '')[:1500]}\n"
        f"Service location: {request.get('service_location_label', '')}\n"
        f"Preferred slot: {request.get('preferred_slot', '')}\n\n"
        "Please confirm whether you can take this appointment and set an ETA."
    )


def build_customer_confirmation_message(request: dict[str, Any]) -> str:
    """Prepare the customer-facing accepted-appointment message."""
    mechanic = request.get("mechanic_name") or "your assigned mechanic"
    eta = request.get("eta_minutes")
    eta_text = f"within {eta} minutes" if eta is not None else "shortly"
    phone = request.get("mechanic_phone") or "not listed"
    fee = request.get("estimated_fee")
    deal = request.get("deal_note")
    text = (
        f"AutoSage appointment confirmed! {mechanic} has accepted your vehicle service request "
        f"and will reach you {eta_text}. Mechanic contact: {phone}."
    )
    if fee:
        text += f" Estimated deal/service fee: {fee}."
    if deal:
        text += f" Note: {deal}"
    return text


SERVICE_TOOLS = {
    "create_service_request": create_service_request,
    "get_service_request": get_service_request,
    "get_service_requests": get_service_requests,
    "update_service_request": update_service_request,
    "build_mechanic_notification_message": build_mechanic_notification_message,
    "build_customer_confirmation_message": build_customer_confirmation_message,
    "build_whatsapp_link": build_whatsapp_link,
}

# LangChain tool wrappers keep service-dispatch capabilities discoverable and
# callable through the same tool abstraction used by the diagnostic agent.
from langchain.tools import tool
from langchain_core.tools import StructuredTool


@tool("service_request_lookup", description="Look up one AutoSage mechanic service request by request ID.")
def service_request_lookup_tool(request_id: str) -> dict[str, Any]:
    return get_service_request(request_id) or {"ok": False, "error": "Request not found."}


@tool("service_request_status", description="List recent AutoSage mechanic service requests, optionally by status.")
def service_request_status_tool(status: str = "") -> list[dict[str, Any]]:
    return get_service_requests(status=status or None, limit=20)


@tool("service_request_confirmation", description="Build the customer-facing confirmation message after a mechanic accepts a request.")
def service_request_confirmation_tool(request_id: str) -> str:
    request = get_service_request(request_id)
    return build_customer_confirmation_message(request or {"id": request_id})


LANGCHAIN_SERVICE_TOOLS: list[StructuredTool] = [
    service_request_lookup_tool,
    service_request_status_tool,
    service_request_confirmation_tool,
]
