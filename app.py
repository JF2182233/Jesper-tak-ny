# app.py
# Run:
#   pip install streamlit shapely plotly pandas
#   streamlit run app.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union


# ============================
# CONFIG (all knobs live here)
# ============================
APP = {
    "page_title": "Roof Estimate",
    "layout": "wide",
    "styles": """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 55%, #0b1324 100%);
            color: #e2e8f0;
        }
        </style>
    """,
    "copy": {
        "title": "Roof Estimate",
        "caption": (
            "Enter your measurements to get a quick material and price estimate. "
            "We’ll confirm final quantities before ordering."
        ),
        "estimate_tab": "Provide estimate",
        "help_tab": "How to measure",
        "help_md": """
**You only need 5 measurements (in meters):**

1. **Roof width** – full width of the roof face along the eaves.
2. **Roof length** – length from eave up to ridge **measured on the roof surface**.
3. **Bump-out start** – distance from the left edge to where the bump-out starts.
4. **Bump-out width** – total width of the bump-out section.
5. **Bump-out depth** – how far “up the roof” the bump-out goes.

If you’re unsure, enter your best guess — this tool is for a quick estimate.
        """,
        "rough_note": "Note: This is a rough estimate. Final quote may change after verification / site check.",
    },
}

PRICING = {
    "materials": {
        "TP 20 (129kr/m2)": 129.0,
        "Stilpanna 118kr/m2": 118.0,
        "Economy metal roofing": 95.0,
    }
}

DEFAULTS = {
    "units": "Meters (recommended)",
    "roof_width": 11.20,
    "roof_length": 4.00,
    "bump_start": 3.00,
    "bump_width": 5.20,
    "bump_depth": 1.50,
    "material_index": 0,
    "sheet_width_mm": 475.0,
    "side_overlap_mm": 0.0,
    "waste_pct": 0.0,
    "direction_label": "Right to left",
}

LIMITS = {
    "roof_width_min": 0.1,
    "roof_length_min": 0.1,
    "bump_min": 0.0,
    "sheet_width_mm": (100.0, 2000.0),
    "side_overlap_mm": (0.0, 200.0),
    "waste_pct": (0.0, 50.0),
}

NUMERICS = {
    "BIG": 1e9,
    "EPS": 1e-6,
    "M_TO_MM": 1000.0,
    "MM2_TO_M2": 1e-6,
}


# ----------------------------
# Data models
# ----------------------------
@dataclass(frozen=True)
class RoofInputs:
    roof_width: float
    roof_length: float
    bump_start: float
    bump_width: float
    bump_depth: float
    units: str

    def to_mm(self) -> "RoofInputsMM":
        if self.units.startswith("Meters"):
            k = NUMERICS["M_TO_MM"]
            return RoofInputsMM(
                roof_width_mm=self.roof_width * k,
                roof_length_mm=self.roof_length * k,
                bump_start_mm=self.bump_start * k,
                bump_width_mm=self.bump_width * k,
                bump_depth_mm=self.bump_depth * k,
            )
        return RoofInputsMM(
            roof_width_mm=self.roof_width,
            roof_length_mm=self.roof_length,
            bump_start_mm=self.bump_start,
            bump_width_mm=self.bump_width,
            bump_depth_mm=self.bump_depth,
        )


@dataclass(frozen=True)
class RoofInputsMM:
    roof_width_mm: float
    roof_length_mm: float
    bump_start_mm: float
    bump_width_mm: float
    bump_depth_mm: float

    @property
    def bump_half_width_mm(self) -> float:
        return self.bump_width_mm / 2.0


@dataclass(frozen=True)
class SheetSettings:
    raw_width_mm: float
    side_overlap_mm: float
    waste_pct: float
    direction: str  # "left_to_right" | "right_to_left"

    @property
    def coverage_width_mm(self) -> float:
        return max(1.0, self.raw_width_mm - self.side_overlap_mm)


@dataclass
class Panel:
    idx: int
    u0: float
    u1: float
    width: float
    left_len: float
    right_len: float
    max_len: float
    note: str = ""


# ----------------------------
# Streamlit setup
# ----------------------------
def setup_page() -> None:
    st.set_page_config(page_title=APP["page_title"], layout=APP["layout"])
    st.markdown(APP["styles"], unsafe_allow_html=True)


# ----------------------------
# Geometry
# ----------------------------
def face_polygon_gable_notch(
    roof_width_mm: float,
    roof_length_mm: float,
    bump_start_mm: float,
    bump_half_width_mm: float,
    bump_depth_mm: float,
) -> Polygon:
    A = roof_width_mm
    H = roof_length_mm
    C = bump_start_mm
    G = bump_half_width_mm
    E = bump_depth_mm

    pts = [
        (0.0, 0.0),
        (0.0, H),
        (A, H),
        (A, 0.0),
        (C + 2 * G, 0.0),
        (C + G, E),
        (C, 0.0),
    ]
    return Polygon(pts)


def _collect_vertical_segments(geom) -> List[Tuple[float, float]]:
    if geom.is_empty:
        return []

    gt = geom.geom_type
    if gt == "LineString":
        ys = [y for _, y in geom.coords]
        return [(min(ys), max(ys))] if ys else []

    if gt == "MultiLineString":
        out: List[Tuple[float, float]] = []
        for ls in geom.geoms:
            out.extend(_collect_vertical_segments(ls))
        return out

    if gt == "GeometryCollection":
        out: List[Tuple[float, float]] = []
        for g in geom.geoms:
            out.extend(_collect_vertical_segments(g))
        return out

    return []


def vertical_spans(poly: Polygon, u: float) -> List[Tuple[float, float]]:
    minx, miny, maxx, maxy = poly.bounds
    big = NUMERICS["BIG"]
    line = LineString([(u, miny - big), (u, maxy + big)])
    return _collect_vertical_segments(poly.intersection(line))


def strip_cut_length(poly: Polygon, u0: float, u1: float) -> float:
    big = NUMERICS["BIG"]
    strip = box(u0, -big, u1, big)
    clipped = poly.intersection(strip)
    if clipped.is_empty:
        return 0.0
    _, miny, _, maxy = clipped.bounds
    return maxy - miny


# ----------------------------
# Panelization
# ----------------------------
def panelize_face(poly: Polygon, coverage_w_mm: float, direction: str) -> List[Panel]:
    minx, _, maxx, _ = poly.bounds
    total_w = maxx - minx
    n = max(1, math.ceil(total_w / coverage_w_mm))

    eps = NUMERICS["EPS"]
    panels: List[Panel] = []

    for i in range(n):
        if direction == "left_to_right":
            u0 = minx + i * coverage_w_mm
            u1 = min(u0 + coverage_w_mm, maxx)
        else:
            u_end = maxx - i * coverage_w_mm
            u0 = max(u_end - coverage_w_mm, minx)
            u1 = u_end

        width = u1 - u0
        max_len = strip_cut_length(poly, u0, u1)

        if max_len <= 0:
            panels.append(Panel(i + 1, u0, u1, width, 0.0, 0.0, 0.0, note="No intersection"))
            continue

        left_spans = vertical_spans(poly, u0 + eps)
        right_spans = vertical_spans(poly, u1 - eps)

        left_len = max((v1 - v0 for v0, v1 in left_spans), default=0.0)
        right_len = max((v1 - v0 for v0, v1 in right_spans), default=0.0)

        panels.append(Panel(i + 1, u0, u1, width, left_len, right_len, max_len))

    return panels


# ----------------------------
# Plotting
# ----------------------------
def _add_polygon_trace(fig: go.Figure, poly: Polygon, name: str) -> None:
    x, y = poly.exterior.xy
    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", name=name))


def plot_face_and_panels(poly: Polygon, panels: Sequence[Panel]) -> go.Figure:
    fig = go.Figure()
    y_top = poly.bounds[3]

    # Panel rectangles (visual aid)
    for p in panels:
        if p.max_len <= 0:
            continue
        y_bottom = y_top - p.max_len
        rect = box(p.u0, y_bottom, p.u1, y_top)
        x, y = rect.exterior.xy
        fig.add_trace(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                fill="toself",
                fillcolor="rgba(30, 144, 255, 0.25)",
                line=dict(color="rgba(30, 144, 255, 0.6)"),
                showlegend=False,
            )
        )

    # Outline (handles polygon or multipolygon, just in case)
    if poly.geom_type == "Polygon":
        _add_polygon_trace(fig, poly, "Face outline")
        bounds_poly = poly
    else:
        for i, p in enumerate(poly.geoms, start=1):
            _add_polygon_trace(fig, p, f"Face outline {i}")
        bounds_poly = unary_union(list(poly.geoms))

    # Guide lines + annotations
    y0, y1 = bounds_poly.bounds[1], bounds_poly.bounds[3]
    for p in panels:
        for x in (p.u0, p.u1):
            fig.add_trace(go.Scatter(x=[x, x], y=[y0, y1], mode="lines", showlegend=False))

        mid = (p.u0 + p.u1) / 2.0
        spans = vertical_spans(poly, mid)
        if spans:
            vmin, vmax = max(spans, key=lambda s: s[1] - s[0])
            fig.add_annotation(x=mid, y=(vmin + vmax) / 2.0, text=f"{p.max_len:.0f}", showarrow=False)

    fig.update_layout(
        xaxis_title="Across roof (mm)",
        yaxis_title="Up roof face (mm)",
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ----------------------------
# Validation & formatting
# ----------------------------
def validate_inputs(mm: RoofInputsMM) -> List[str]:
    errors: List[str] = []

    if mm.roof_width_mm <= 0 or mm.roof_length_mm <= 0:
        errors.append("Roof width and roof length must be greater than 0.")

    if mm.bump_start_mm < 0 or mm.bump_width_mm < 0 or mm.bump_depth_mm < 0:
        errors.append("Bump-out values cannot be negative.")

    if mm.bump_start_mm + mm.bump_width_mm > mm.roof_width_mm:
        errors.append("Bump-out start + bump-out width must be within the roof width.")

    if mm.bump_depth_mm > mm.roof_length_mm:
        errors.append("Bump-out depth cannot be greater than the roof length.")

    return errors


def format_sek(x: float) -> str:
    return f"{x:,.0f} SEK".replace(",", " ")


# ----------------------------
# UI
# ----------------------------
def render_help_tab() -> None:
    st.subheader("How to measure (simple)")
    st.markdown(APP["copy"]["help_md"])


def render_estimate_tab() -> None:
    left, right = st.columns([1.1, 1.2], gap="large")

    with left:
        st.subheader("1) Roof measurements")

        with st.form("estimate_form", clear_on_submit=False):
            units = st.radio("Units", ["Meters (recommended)", "Millimeters"], horizontal=True)

            roof_width = st.number_input(
                "Roof width (along the eaves)",
                min_value=LIMITS["roof_width_min"],
                value=DEFAULTS["roof_width"],
                step=0.1,
                help="Total width of this roof face along the bottom edge (eaves).",
            )
            roof_length = st.number_input(
                "Roof length (eave → ridge) on the roof surface",
                min_value=LIMITS["roof_length_min"],
                value=DEFAULTS["roof_length"],
                step=0.1,
                help="Measure along the roof surface from the eave up to the ridge.",
            )

            st.markdown("**Bump-out (utbyggnad) position**")
            bump_start = st.number_input(
                "Distance from left edge to bump-out start",
                min_value=LIMITS["bump_min"],
                value=DEFAULTS["bump_start"],
                step=0.05,
                help="From the left edge of the roof face, measure to where the bump-out begins.",
            )
            bump_width = st.number_input(
                "Bump-out width (total)",
                min_value=LIMITS["bump_min"],
                value=DEFAULTS["bump_width"],
                step=0.05,
                help="Total width of the bump-out section.",
            )
            bump_depth = st.number_input(
                "Bump-out depth (how far up the roof it goes)",
                min_value=LIMITS["bump_min"],
                value=DEFAULTS["bump_depth"],
                step=0.05,
                help="How far the bump-out reaches up into the roof face.",
            )

            st.divider()
            st.subheader("2) Material choice")

            materials = list(PRICING["materials"].keys())
            material = st.selectbox("Roofing material", materials, index=DEFAULTS["material_index"])
            price_per_m2 = PRICING["materials"][material]

            with st.expander("Advanced sheet settings (optional)"):
                lo, hi = LIMITS["sheet_width_mm"]
                raw_w = st.number_input("Sheet width (mm)", lo, hi, DEFAULTS["sheet_width_mm"], 1.0)

                lo, hi = LIMITS["side_overlap_mm"]
                side_lap = st.number_input("Side overlap (mm)", lo, hi, DEFAULTS["side_overlap_mm"], 1.0)

                lo, hi = LIMITS["waste_pct"]
                waste_pct = st.number_input("Waste %", lo, hi, DEFAULTS["waste_pct"], 0.5)

                direction_label = st.selectbox(
                    "Panel layout direction",
                    ["Right to left", "Left to right"],
                    index=0 if DEFAULTS["direction_label"] == "Right to left" else 1,
                )

            direction = "right_to_left" if direction_label == "Right to left" else "left_to_right"
            submitted = st.form_submit_button("Calculate estimate")

        if not submitted:
            st.info("Fill in the measurements and press **Calculate estimate**.")
            st.stop()

        # Convert + validate
        raw = RoofInputs(roof_width, roof_length, bump_start, bump_width, bump_depth, units)
        mm = raw.to_mm()

        errors = validate_inputs(mm)
        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        # Build polygon
        poly = face_polygon_gable_notch(
            roof_width_mm=mm.roof_width_mm,
            roof_length_mm=mm.roof_length_mm,
            bump_start_mm=mm.bump_start_mm,
            bump_half_width_mm=mm.bump_half_width_mm,
            bump_depth_mm=mm.bump_depth_mm,
        )

        if not poly.is_valid:
            st.error("These measurements produce an invalid shape. Double-check your numbers.")
            st.stop()

        # Panelize + compute estimate
        sheet = SheetSettings(raw_w, side_lap, waste_pct, direction)
        panels = panelize_face(poly, coverage_w_mm=sheet.coverage_width_mm, direction=sheet.direction)

        area_m2 = poly.area * NUMERICS["MM2_TO_M2"]
        area_m2_waste = area_m2 * (1.0 + sheet.waste_pct / 100.0)
        cost = area_m2_waste * price_per_m2

        st.subheader("Estimate summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated roof face area", f"{area_m2:.2f} m²")
        c2.metric("Estimated area incl. waste", f"{area_m2_waste:.2f} m²", delta=f"+{sheet.waste_pct:.1f}%")
        c3.metric("Estimated material cost", format_sek(cost))

        st.caption(APP["copy"]["rough_note"])

    with right:
        st.subheader("Preview (for confirmation)")
        fig = plot_face_and_panels(poly, panels)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Installer details (cut list / panel data)"):
            df = pd.DataFrame(
                [
                    {
                        "Panel #": p.idx,
                        "U start (mm)": round(p.u0, 1),
                        "U end (mm)": round(p.u1, 1),
                        "Width (mm)": round(p.width, 1),
                        "Cut length (mm)": round(p.max_len, 1),
                        "Left length (mm)": round(p.left_len, 1),
                        "Right length (mm)": round(p.right_len, 1),
                        "Note": p.note,
                    }
                    for p in panels
                ]
            )
            st.dataframe(df, use_container_width=True)

            longest = max((p.max_len for p in panels), default=0.0)
            st.write(f"**Number of panels:** {len(panels)}")
            st.write(f"**Longest required panel length (approx):** {longest:.0f} mm")


# ----------------------------
# App entry
# ----------------------------
def main() -> None:
    setup_page()

    st.title(APP["copy"]["title"])
    st.caption(APP["copy"]["caption"])

    tab_est, tab_help = st.tabs([APP["copy"]["estimate_tab"], APP["copy"]["help_tab"]])
    with tab_help:
        render_help_tab()
    with tab_est:
        render_estimate_tab()


if __name__ == "__main__":
    main()
