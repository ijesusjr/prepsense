"""
app.py
-------
PrepSense Streamlit dashboard — single-page layout.

Structure:
    Header          — title + refresh button (full width)
    Row 1           — Map (left) | Risk Indexes (right)
    Section header  — EMERGENCY KIT MANAGEMENT
    Row 2           — Kit Status + Kit Editor▶ (left) | Action List scrollable (right)
    Row 3           — Chat (left) | Scenario Simulator (right)
"""

import os
import time
import requests
import streamlit as st
from core.regions import is_supported

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8000")

LEVEL_COLORS = {
    "LOW":      "#2ecc71",
    "MINIMAL":  "#2ecc71",
    "ROUTINE":  "#2ecc71",
    "MEDIUM": "#f39c12",
    "HIGH":     "#e74c3c",
    "CRITICAL": "#8e44ad",
}

URGENCY_COLORS = {
    "IMMEDIATE": "#e74c3c",
    "SOON":      "#f39c12",
    "ROUTINE":   "#95a5a6",
}

CATEGORY_ICONS = {
    "WEATHER":    "🌤",
    "COMBINED":   "⚡",
    "HEALTH_KIT": "🦠💊",
    "EXPIRY":     "⏰",
    "KIT_GAP":    "📦",
    "HEALTH":     "🦠",
    "GEO":        "⚔",
}

EVENT_TYPES = {
    "General emergency": "general",
    "Power outage":      "power_outage",
    "Flood":             "flood",
    "Earthquake":        "earthquake",
    "Heat wave":         "heat_wave",
}

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path: str) -> dict:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_post(path: str, body: dict) -> dict:
    try:
        r = requests.post(f"{API_URL}{path}", json=body, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_put(path: str, body: dict) -> dict:
    try:
        r = requests.put(f"{API_URL}{path}", json=body, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def level_badge(level: str) -> str:
    color = LEVEL_COLORS.get(level, "#95a5a6")
    return (f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.8em;font-weight:bold">{level}</span>')

def score_gauge(label: str, score: int, max_score: int, level: str):
    color = LEVEL_COLORS.get(level, "#95a5a6")
    pct   = int(score / max_score * 100)
    badge = level_badge(level)
    st.markdown(f"""
    <div style="background:#1e1e2e;border-radius:8px;padding:14px 18px;margin:4px 0">
      <div style="font-size:1.2em;font-weight:bold;color:#aaa;margin-bottom:4px">{label}</div>
      <div style="font-size:1.6em;font-weight:bold;color:{color}">{score}
        <span style="font-size:0.55em;color:#888">/ {max_score}</span>
      </div>
      <div style="background:#333;border-radius:4px;height:6px;margin:8px 0">
        <div style="background:{color};width:{pct}%;height:6px;border-radius:4px"></div>
      </div>
      {badge}
    </div>
    """, unsafe_allow_html=True)

def alert_card_html(alert: dict) -> str:
    color  = URGENCY_COLORS.get(alert["urgency"], "#95a5a6")
    icon   = CATEGORY_ICONS.get(alert["category"], "ℹ")
    return (
        f'<div style="border-left:3px solid {color};padding:8px 12px;margin:6px 0;'
        f'background:#1a1a2e;border-radius:0 6px 6px 0">'
        f'<div style="font-size:0.75em;color:{color};font-weight:bold;margin-bottom:2px">'
        f'{icon} {alert["category"]} · {alert["urgency"]} · score {alert["priority_score"]}'
        f'</div>'
        f'<div style="font-size:0.9em">{alert["message"]}</div>'
        f'</div>'
    )

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PrepSense",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 1rem; }
  /* scrollable alert container */
  .alert-scroll {
      max-height: 420px;
      overflow-y: auto;
      padding-right: 4px;
  }
  /* chat column scroll */
  .chat-history {
      overflow-y: auto;
      padding: 4px 0;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

risk_data  = api_get("/risk")
kit_data   = api_get("/kit")
alert_data = api_get("/alerts")

api_ok = "error" not in risk_data

# ---------------------------------------------------------------------------
# ── HEADER ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

col_title, col_meta, col_refresh = st.columns([3, 2, 1])
with col_title:
    st.markdown("# 🛡 PrepSense")
    st.markdown("*AI-powered emergency preparedness copilot*")
with col_meta:
    if api_ok:
        meta   = risk_data.get("meta", {})
        age    = meta.get("age_minutes", "?")
        stale  = meta.get("stale", False)
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            f"🕐 Updated {age} min ago  |  "
            f"{'⚠ stale' if stale else '✓ fresh'}"
        )
with col_refresh:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh", use_container_width=True):
        api_get("/refresh")
        st.toast("Signals refreshing...", icon="🔄")
        time.sleep(1)
        st.rerun()

if not api_ok:
    st.error(f"⚠ Cannot reach API at {API_URL} — {risk_data.get('error')}")
    st.info("Start the API with: `uvicorn api.main:app --reload`")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# ── ROW 1: MAP | RISK INDEXES ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

col_map, col_risk = st.columns([0.55, 0.45])

# ── Map ──────────────────────────────────────────────────────────────────────
with col_map:
    st.markdown("### 📍 Location")

    import folium
    from streamlit_folium import st_folium

    if "user_lat" not in st.session_state:
        st.session_state.user_lat     = float(os.getenv("LAT",  "41.3851"))
        st.session_state.user_lon     = float(os.getenv("LON",  "2.1734"))
        st.session_state.user_city    = os.getenv("CITY", "Barcelona")
        st.session_state.user_country = ""

    lat = st.session_state.user_lat
    lon = st.session_state.user_lon

    m = folium.Map(
        location=[54, 15],       # centre of Europe — pin still goes on user location
        zoom_start=4,
        tiles="CartoDB positron",
        min_zoom=3,
        max_zoom=12,
    )
    m.fit_bounds([[34.0, -12.0], [72.0, 45.0]])

    folium.Marker(
        location=[lat, lon],
        popup=st.session_state.user_city,
        tooltip="Your location",
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)

    map_data = st_folium(m, width="100%", height=600,
                         returned_objects=["last_clicked"])

    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        city    = f"{clicked_lat:.2f}, {clicked_lon:.2f}"
        country = ""
        try:
            geo      = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"lat": clicked_lat, "lon": clicked_lon,
                        "format": "json", "zoom": 10},
                headers={"User-Agent": "PrepSense/1.0",
                         "Accept-Language": "en"},
                timeout=5,
            ).json()
            geo_addr = geo.get("address", {})
            city     = (geo_addr.get("city") or geo_addr.get("town")
                        or geo_addr.get("village")
                        or geo.get("display_name", "").split(",")[0])
            country  = geo_addr.get("country", "")
        except Exception:
            pass

        # Support warning
        if country and not is_supported(country):
            st.warning(
                f"⚠ **{country}** not in supported region list — "
                f"regional risk uses default EU neighbours."
            )
        elif country:
            st.success(f"✓ **{country}** fully supported.")

        loc_col, btn_col = st.columns([3, 1])
        with loc_col:
            st.info(f"📍 **{city}**, {country} "
                    f"({clicked_lat:.3f}, {clicked_lon:.3f})")
        with btn_col:
            if st.button("✓ Set", type="primary", use_container_width=True):
                result = api_put("/location", {
                    "lat": clicked_lat, "lon": clicked_lon,
                    "city": city, "country": country,
                })
                if "error" not in result:
                    st.session_state.user_lat     = clicked_lat
                    st.session_state.user_lon     = clicked_lon
                    st.session_state.user_city    = city
                    st.session_state.user_country = country
                    triggers = result.get("triggers", ["weather"])
                    st.toast(
                        f"📍 {city} — {', '.join(triggers)} updating...",
                        icon="📍",
                    )
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(result["error"])

    # Footer
    country_str = st.session_state.get("user_country", "")
    city_str    = st.session_state.user_city
    st.markdown(
        f'<div style="font-size:0.82em;color:#aaa;margin-top:4px">'
        f'📍 <b style="color:#ddd">{city_str}'
        + (f', {country_str}' if country_str else '')
        + f'</b> · {lat:.4f}, {lon:.4f}'
        f'<br><span style="font-size:0.9em">Weather and regional signals refresh '
        f'automatically after location change. '
        f'Health signal is EU-wide and does not change with location.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Risk indexes ─────────────────────────────────────────────────────────────
with col_risk:
    st.markdown("### Risk Signals")

    w = risk_data["weather"]
    r = risk_data["regional"]
    h = risk_data["health"]

    score_gauge("🌤 Weather Risk", w["score"], 100, w["level"])
    if w["breakdown"]["alert_severity"] > 0:
        st.caption(f"Active alert · severity {w['breakdown']['alert_severity']}")

    with st.expander("📊 Weather score breakdown", expanded=False):
        bd = w["breakdown"]
        items_bd = [
            ("⚡ Active alert",    bd["alert_severity"],   60,  "From OWM national alert tags (Minor/Moderate/Severe/Extreme)"),
            ("🌡 Condition",       bd["weather_severity"], 40,  "From OWM weather ID — thunderstorm, snow, fog, rain, etc."),
            ("💨 Wind bonus",      bd["wind_bonus"],       15,  "Wind speed > 10 m/s adds points"),
            ("🌧 Rain bonus",      bd["rain_bonus"],       10,  "Rainfall > 2 mm/h adds points"),
        ]
        for label, score, max_s, note in items_bd:
            color = "#e74c3c" if score >= max_s * 0.6 else "#f39c12" if score > 0 else "#555"
            pct   = int(score / max_s * 100) if max_s > 0 else 0
            st.markdown(
                f'<div style="margin:6px 0">'
                f'<div style="font-size:0.8em;color:#aaa">{label} '
                f'<span style="color:{color};font-weight:bold">+{score}</span>'
                f'<span style="color:#666"> / {max_s}</span></div>'
                f'<div style="background:#333;border-radius:3px;height:4px;margin:3px 0">'
                f'<div style="background:{color};width:{pct}%;height:4px;border-radius:3px"></div>'
                f'</div>'
                f'<div style="font-size:0.72em;color:#666">{note}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #333;'
            f'font-size:0.85em">Total: <b style="color:white">{w["score"]}</b>/100 → '
            f'<b>{w["level"]}</b></div>',
            unsafe_allow_html=True,
        )
    with st.expander("ℹ About this risk", expanded=False):
        st.markdown("""
                    <div style="font-size:0.8em">
                    
        **🌤 Weather Risk** `0–100`
        Measures active national weather alerts and current conditions
        (wind speed, rainfall, weather type) at your location.
        - **Source:** OpenWeatherMap One Call 3.0
        - **Updated:** Every 60 minutes
        - **Location-specific:** Yes — changes when you move the pin</div>
    """, unsafe_allow_html=True)

    score_gauge("⚔ Regional Risk", r["score"], 30, r["level"])
    st.caption(f"Trend: {r['trend']} · {r['country']}")
    with st.expander("ℹ About this risk", expanded=False):
        st.markdown("""
                    <div style="font-size:0.8em">
                    
        **⚔ Regional Risk** `0–30`
        Combines natural disaster alerts and humanitarian crisis reports
        in your country and neighbouring countries.
        - **Sources:** GDACS (UN/EC — earthquakes, floods, wildfires) +
        ReliefWeb (UN OCHA — conflict, displacement, humanitarian crises)
        - **Updated:** Every 7 days
        - **Location-specific:** Yes — neighbours change with your country</div>
    """, unsafe_allow_html=True)

    score_gauge("🦠 Health Risk", h["score"], 50, h["level"])
    if h["top_threats"]:
        st.caption(f"Active: {', '.join(h['top_threats'][:2])}")
    else:
        st.caption("EU-wide signal · not location-specific")

    with st.expander("ℹ About this risk", expanded=False):
        st.markdown("""
                    <div style="font-size:0.8em">
                    
        **🦠 Health Risk** `0–50`
        Monitors active communicable disease threats across the EU/EEA,
        based on the weekly ECDC bulletin (CDTR).
        - **Source:** ECDC Communicable Disease Threats Report
        - **Updated:** Every 7 days (published Thursdays)
        - **Location-specific:** No — EU-wide signal, same for all EU countries</div>
    """, unsafe_allow_html=True)


st.divider()

# ---------------------------------------------------------------------------
# ── ROW 2: KIT STATUS + KIT EDITOR | ACTION LIST ────────────────────────────
# ---------------------------------------------------------------------------

st.markdown("### Emergency Kit Management")

col_kit, col_alerts = st.columns([0.5, 0.5])

# ── Kit status ───────────────────────────────────────────────────────────────
with col_kit:
    summary  = kit_data.get("summary", {})
    gaps     = kit_data.get("gaps", [])
    expiring = kit_data.get("expiring", [])
    gap_score = summary.get("gap_score", 0)

    household = kit_data.get("household_size", 1)
    st.markdown(
        f"**📦 Kit Status** — "
        f"`{summary.get('total_gaps', 0)} gaps · score {gap_score}/100 · "
        f"{household} person(s)`"
    )

    if not gaps and not expiring:
        st.success("✓ Kit complete — all items at recommended levels.")
    else:
        if gaps:
            for g in gaps:
                to_buy = g["recommended"] - g["current"]
                color  = "#e74c3c" if g["priority"] == "HIGH" else "#f39c12"
                pct    = g["gap_pct"]
                bar    = int(pct / 5)
                st.markdown(
                    f'<div style="margin:5px 0;font-size:0.85em">'
                    f'<div style="margin-bottom:2px">'
                    f'<b>{g["name"]}</b> '
                    f'<span style="color:{color}">need to buy {to_buy:.1f} {g["unit"]}</span>'
                    f'<span style="color:#666;font-size:0.85em"> '
                    f'(have {g["current"]:.1f} / need {g["recommended"]:.1f})</span>'
                    f'</div>'
                    f'<div style="background:#333;border-radius:3px;height:4px">'
                    f'<div style="background:{color};width:{pct}%;height:4px;border-radius:3px"></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        if expiring:
            st.markdown("**⏰ Expiring soon:**")
            for e in expiring:
                color = "#e74c3c" if e["urgency"] == "CRITICAL" else "#f39c12"
                st.markdown(
                    f'<span style="color:{color}">⏰ {e["name"]}: '
                    f'{e["days_remaining"]}d ({e["expiry_date"]})</span>',
                    unsafe_allow_html=True,
                )

    # ── Kit editor (expandable) ───────────────────────────────────────────────
    st.markdown("")
    with st.expander("✏ Edit Kit", expanded=False):
        items = kit_data.get("items", [])
        if not items:
            st.warning("No kit items found.")
        else:
            # Household size
            if "household_size" not in st.session_state:
                st.session_state.household_size = int(os.getenv("HOUSEHOLD_SIZE", "1"))

            st.session_state.household_size = st.number_input(
                "👥 People in household",
                min_value=1, max_value=10,
                value=st.session_state.household_size,
                key="household_size_input",
                help="Scales recommended quantities for water, food, medication, "
                     "hand sanitizer, and cash.",
            )
            n = st.session_state.household_size

            # Items whose EU recommendation scales with household size
            PER_PERSON_ITEMS = {
                "Drinking water":      True,   # 3L/person/day × 3 days = 9L
                "Non-perishable food": True,   # 3 days/person
                "Regular medication":  True,   # 7 days/person
                "Hand sanitizer":      True,   # 1 unit/person
                "Cash":                True,   # €70/adult
            }

            st.divider()

            # Collect edits in session state
            if "kit_edits" not in st.session_state:
                st.session_state.kit_edits = {
                    i["name"]: {
                        "quantity":    i["quantity"],
                        "expiry_date": i.get("expiry_date") or "",
                    }
                    for i in items
                }

            categories = {}
            for item in items:
                categories.setdefault(item["category"], []).append(item)

            for cat, cat_items in sorted(categories.items()):
                st.markdown(f"**{cat.title()}**")
                for item in cat_items:
                    name       = item["name"]
                    scales     = PER_PERSON_ITEMS.get(name, False)
                    base_rec   = item["eu_recommended"]
                    scaled_rec = base_rec * n if scales else base_rec
                    rec_label  = (
                        f"{scaled_rec:.0f} {item['unit']} "
                        f"({base_rec:.0f} × {n} people)"
                        if scales and n > 1
                        else f"{scaled_rec:.0f} {item['unit']}"
                    )

                    c1, c2, c3 = st.columns([2.5, 1.2, 1.5])
                    with c1:
                        scale_icon = "👥" if scales else ""
                        st.markdown(
                            f'<div style="padding-top:8px;font-size:0.9em">'
                            f'{scale_icon} {name}'
                            f'<span style="color:#888;font-size:0.8em"> '
                            f'(rec: {rec_label})</span></div>',
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.session_state.kit_edits[name]["quantity"] = st.number_input(
                            "Qty", value=float(item["quantity"]),
                            min_value=0.0, step=0.5,
                            key=f"qty_{name}",
                            label_visibility="collapsed",
                        )
                    with c3:
                        st.session_state.kit_edits[name]["expiry_date"] = st.text_input(
                            "Expiry",
                            value=item.get("expiry_date") or "",
                            key=f"exp_{name}",
                            label_visibility="collapsed",
                            placeholder="YYYY-MM-DD",
                        )

            st.markdown("")
            if st.button("💾 Save All", type="primary", use_container_width=True):
                # Update household size in API first
                hh_result = api_put("/household", {"size": n})
                if "error" in hh_result:
                    st.error(f"Household update failed: {hh_result['error']}")
                
                # Then save kit items
                errors = []
                for name, edits in st.session_state.kit_edits.items():
                    payload = {"quantity": edits["quantity"]}
                    if edits["expiry_date"]:
                        payload["expiry_date"] = edits["expiry_date"]
                    result = api_put(f"/kit/{name}", payload)
                    if "error" in result:
                        errors.append(f"{name}: {result['error']}")
                if errors:
                    st.error("\n".join(errors))
                else:
                    st.toast(f"✓ Kit saved for {n} person(s)", icon="✅")
                    time.sleep(0.5)
                    st.rerun()

# ── Action list (scrollable) ──────────────────────────────────────────────────
with col_alerts:
    alerts = alert_data.get("alerts", [])
    st.markdown(f"**🚨 Action List** — `{len(alerts)} items`")

    if not alerts:
        st.success("✓ No active alerts — kit looks good for current risk levels.")
    else:
        # Build HTML for all alerts and wrap in scrollable div
        alerts_html = "".join(alert_card_html(a) for a in alerts)
        st.markdown(
            f'<div class="alert-scroll">{alerts_html}</div>',
            unsafe_allow_html=True,
        )
        if len(alerts) > 5:
            st.caption(f"↕ Scroll to see all {len(alerts)} alerts")

st.divider()

# ---------------------------------------------------------------------------
# ── ROW 3: CHAT | SCENARIO SIMULATOR ────────────────────────────────────────
# ---------------------------------------------------------------------------

col_chat, col_scenario = st.columns([0.5, 0.5])

# ── Chat ─────────────────────────────────────────────────────────────────────
with col_chat:
    st.markdown("### 💬 Chat")
    st.markdown(
        "<small>Ask about your kit, current risks, or preparedness advice.</small>",
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Message history — grows naturally
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for s in msg["sources"]:
                        st.caption(f"• {s}")

    # Chat input pinned inside this column
    if question := st.chat_input("Ask a question...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = api_post("/chat", {"question": question})

            if "error" in result:
                answer, sources = f"Error: {result['error']}", []
                st.error(answer)
            else:
                answer   = result.get("answer", "No answer returned.")
                sources  = result.get("sources", [])
                intent   = result.get("intent", "")
                fallback = result.get("fallback", False)

                st.markdown(answer)

                badge_color = "#555" if fallback else "#2980b9"
                st.markdown(
                    f'<span style="font-size:0.75em;color:#888">Intent: '
                    f'<span style="background:{badge_color};color:white;'
                    f'padding:1px 6px;border-radius:3px">{intent}</span></span>',
                    unsafe_allow_html=True,
                )
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        for s in sources:
                            st.caption(f"• {s}")

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources,
        })

    if st.session_state.messages:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

# ── Scenario simulator ────────────────────────────────────────────────────────
with col_scenario:
    st.markdown("### ⚙ Scenario Simulator")
    st.markdown(
        "<small>Estimate how long your kit lasts in an emergency.</small>",
        unsafe_allow_html=True,
    )

    event_label    = st.selectbox("Event type", list(EVENT_TYPES.keys()),
                                  key="sc_event")
    event_type     = EVENT_TYPES[event_label]
    duration_hours = st.slider("Duration (hours)", 12, 168, 72, step=12,
                               key="sc_duration")
    people         = st.number_input("People in household", 1, 10, 1,
                                     key="sc_people")

    if st.button("▶ Run Scenario", use_container_width=True,
                 type="primary", key="sc_run"):
        with st.spinner("Calculating..."):
            result = api_post("/scenario", {
                "event_type":     event_type,
                "duration_hours": duration_hours,
                "people":         people,
            })

        if "error" in result:
            st.error(result["error"])
        else:
            pct   = result["survival_pct"]
            color = (
                LEVEL_COLORS["HIGH"]     if pct < 40 else
                LEVEL_COLORS["MEDIUM"] if pct < 75 else
                LEVEL_COLORS["LOW"]
            )
            st.markdown(f"""
            <div style="background:#1e1e2e;border-radius:8px;
                        padding:14px 18px;margin:8px 0">
              <div style="font-size:0.9em;color:#aaa">
                {event_label} · {duration_hours}h · {people} person(s)
              </div>
              <div style="font-size:2.8em;font-weight:bold;color:{color}">
                {pct}%
              </div>
              <div style="color:#aaa;font-size:0.85em">scenario coverage</div>
              <div style="background:#333;border-radius:4px;
                          height:7px;margin:8px 0">
                <div style="background:{color};width:{pct}%;
                            height:7px;border-radius:4px"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("💧 Water", f"{result['water_hours']:.0f}h")
            m2.metric("🍞 Food",  f"{result['food_hours']:.0f}h")
            m3.metric("📻 Comms", "✓" if result["comms_ok"]  else "✗")
            m4.metric("💊 Meds",  "✓" if result["meds_ok"]   else "✗")

            if result["critical_gaps"]:
                st.markdown("**⚠ Critical shortfalls:**")
                for g in result["critical_gaps"]:
                    st.error(g)

            if result["recommendations"]:
                st.markdown("**Recommendations:**")
                for rec in result["recommendations"]:
                    st.info(f"• {rec}")

            st.caption(result["narrative"])
