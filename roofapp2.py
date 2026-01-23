# app.py
# Run:
#   pip install streamlit shapely plotly pandas
#   streamlit run app.py

import math
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass
from shapely.geometry import Polygon, LineString, box
from shapely.ops import unary_union

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Roof Estimate", layout="wide")

# ----------------------------
# Global styling
# ----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 55%, #0b1324 100%);
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Face template (your same geometry)
# ----------------------------

def face_polygon_gable_notch(A_mm, H_mm, C_mm, G_mm, E_mm):
    """
    Polygon for ONE roof face with a triangular notch along the bottom edge.
    Units: mm.

    (0,0) -> (0,H) -> (A,H) -> (A,0) -> (C+2G,0) -> (C+G,E) -> (C,0) -> (0,0)
    """
    pts = [
        (0, 0),
        (0, H_mm),
        (A_mm, H_mm),
        (A_mm, 0),
        (C_mm + 2 * G_mm, 0),
        (C_mm + G_mm, E_mm),
        (C_mm, 0),
        (0, 0),
    ]
    return Polygon(pts)

# ----------------------------
# Robust helpers
# ----------------------------

def _collect_segments_from_intersection(geom):
    segments = []
    if geom.is_empty:
        return segments

    gt = geom.geom_type
    if gt == "LineString":
        ys = [y for _, y in geom.coords]
        if ys:
            segments.append((min(ys), max(ys)))
    elif gt == "MultiLineString":
        for ls in geom.geoms:
            segments.extend(_collect_segments_from_intersection(ls))
    elif gt == "GeometryCollection":
        for g in geom.geoms:
            segments.extend(_collect_segments_from_intersection(g))
    return segments

def vertical_spans(poly: Polygon, u: float):
    """
    Return (vmin, vmax) of intersection with vertical line u=const.
    For this roof face shape, it should normally be one continuous segment.
    """
    minx, miny, maxx, maxy = poly.bounds
    line = LineString([(u, miny - 1e6), (u, maxy + 1e6)])

    inter = poly.intersection(line)
    segments = _collect_segments_from_intersection(inter)

    if not segments:
        return []

    return segments

def strip_cut_length(poly: Polygon, u0: float, u1: float) -> float:
    """Return vertical span of roof inside [u0, u1] strip for supplier cut length."""
    big = 1e9
    strip = box(u0, -big, u1, big)
    clipped = poly.intersection(strip)
    if clipped.is_empty:
        return 0.0
    miny, maxy = clipped.bounds[1], clipped.bounds[3]
    return maxy - miny

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

def panelize_face(poly: Polygon, coverage_w: float, direction="left_to_right"):
    """
    Panelize the face into strips of width=coverage_w (mm).
    Returns list[Panel] with left/right edge lengths and max length in strip.
    """
    minx, miny, maxx, maxy = poly.bounds
    total_w = maxx - minx
    n = math.ceil(total_w / coverage_w)

    panels = []

    for i in range(n):
        if direction == "left_to_right":
            u0 = minx + i * coverage_w
            u1 = min(u0 + coverage_w, maxx)
        else:
            u_end = maxx - i * coverage_w
            u0 = max(u_end - coverage_w, minx)
            u1 = u_end

        width = u1 - u0

        # Candidate u positions where the span can change (strip edges + vertex x inside strip)
        verts_u = [x for x, _ in poly.exterior.coords]
        cand = sorted(set([u0, u1] + [x for x in verts_u if u0 < x < u1]))

        spans = []
        for u in cand:
            u_spans = vertical_spans(poly, u)
            spans.extend(u_spans)

        if not spans:
            panels.append(Panel(i+1, u0, u1, width, 0, 0, 0, note="No intersection"))
            continue

        eps = 1e-6
        left_spans = vertical_spans(poly, u0 + eps)
        right_spans = vertical_spans(poly, u1 - eps)

        # Clip to strip to avoid vertex/point intersections that under-report span.
        max_len = strip_cut_length(poly, u0, u1)
        left_len = max((v1 - v0 for v0, v1 in left_spans), default=0)
        right_len = max((v1 - v0 for v0, v1 in right_spans), default=0)

        panels.append(Panel(
            idx=i+1,
            u0=u0, u1=u1, width=width,
            left_len=left_len,
            right_len=right_len,
            max_len=max_len,
            note=""
        ))

    return panels

def plot_face_and_panels(poly, panels):
    fig = go.Figure()
    y_top = poly.bounds[3]

    # Panel rectangles (ordered blanks)
    for p in panels:
        if p.max_len <= 0:
            continue
        y_bottom = y_top - p.max_len
        rect = box(p.u0, y_bottom, p.u1, y_top)
        x, y = rect.exterior.xy
        fig.add_trace(go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines",
            fill="toself",
            fillcolor="rgba(30, 144, 255, 0.25)",
            line=dict(color="rgba(30, 144, 255, 0.6)"),
            showlegend=False,
        ))

    # Outline
    def add_poly(p, name):
        x, y = p.exterior.xy
        fig.add_trace(go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines",
            name=name
        ))

    if poly.geom_type == "Polygon":
        add_poly(poly, "Face outline")
        bounds_poly = poly
    else:
        for i, p in enumerate(poly.geoms, start=1):
            add_poly(p, f"Face outline {i}")
        bounds_poly = unary_union(list(poly.geoms))

    # Panel guide lines + numbers
    y0, y1 = bounds_poly.bounds[1], bounds_poly.bounds[3]
    for p in panels:
        fig.add_trace(go.Scatter(
            x=[p.u0, p.u0],
            y=[y0, y1],
            mode="lines",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[p.u1, p.u1],
            y=[y0, y1],
            mode="lines",
            showlegend=False
        ))

        mid = (p.u0 + p.u1) / 2
        spans = vertical_spans(poly, mid)
        if spans:
            vmin, vmax = max(spans, key=lambda span: span[1] - span[0])
            fig.add_annotation(
                x=mid, y=(vmin + vmax) / 2,
                text=f"{p.max_len:.0f}",
                showarrow=False
            )

    fig.update_layout(
        xaxis_title="Across roof (mm)",
        yaxis_title="Up roof face (mm)",
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

# ----------------------------
# Customer-facing UI
# ----------------------------

st.title("Roof Estimate")
st.caption("Enter your measurements to get a quick material and price estimate. We’ll confirm final quantities before ordering.")

tab_est, tab_help = st.tabs(["Provide estimate", "How to measure"])

with tab_help:
    st.subheader("How to measure (simple)")
    st.markdown(
        """
**You only need 5 measurements (in meters):**

1. **Roof width** – full width of the roof face along the eaves.
2. **Roof length** – length from eave up to ridge **measured on the roof surface**.
3. **Bump-out start** – distance from the left edge to where the bump-out starts.
4. **Bump-out width** – total width of the bump-out section.
5. **Bump-out depth** – how far “up the roof” the bump-out goes.

If you’re unsure, enter your best guess — this tool is for a quick estimate.
        """
    )

with tab_est:
    left, right = st.columns([1.1, 1.2], gap="large")

    with left:
        st.subheader("1) Roof measurements")

        with st.form("estimate_form", clear_on_submit=False):
            units = st.radio("Units", ["Meters (recommended)", "Millimeters"], horizontal=True)

            roof_width = st.number_input(
                "Roof width (along the eaves)",
                min_value=0.1,
                value=11.20,
                step=0.1,
                help="Total width of this roof face along the bottom edge (eaves)."
            )
            roof_length = st.number_input(
                "Roof length (eave → ridge) on the roof surface",
                min_value=0.1,
                value=4.00,
                step=0.1,
                help="Measure along the roof surface from the eave up to the ridge."
            )

            st.markdown("**Bump-out (utbyggnad) position**")
            bump_start = st.number_input(
                "Distance from left edge to bump-out start",
                min_value=0.0,
                value=3.00,
                step=0.05,
                help="From the left edge of the roof face, measure to where the bump-out begins."
            )
            bump_width = st.number_input(
                "Bump-out width (total)",
                min_value=0.0,
                value=5.20,
                step=0.05,
                help="Total width of the bump-out section."
            )
            bump_depth = st.number_input(
                "Bump-out depth (how far up the roof it goes)",
                min_value=0.0,
                value=1.50,
                step=0.05,
                help="How far the bump-out reaches up into the roof face."
            )

            st.divider()
            st.subheader("2) Material choice")

            material = st.selectbox(
                "Roofing material",
                ["TP 20 (129kr/m2)", "Stilpanna 118kr/m2", "Economy metal roofing"],
                index=0
            )

            # You can later replace these with your friend's real pricing rules/catalog
            material_prices = {
                "TP 20 (129kr/m2)": 129.0,
                "Stilpanna 118kr/m2": 118.0,
                "Economy metal roofing": 95.0,
            }
            price_per_m2 = material_prices[material]

            with st.expander("Advanced sheet settings (optional)"):
                raw_w = st.number_input("Sheet width (mm)", 100.0, 2000.0, 475.0, 1.0)
                side_lap = st.number_input("Side overlap (mm)", 0.0, 200.0, 0.0, 1.0)
                waste_pct = st.number_input("Waste %", 0.0, 50.0, 0.0, 0.5)
                direction_label = st.selectbox(
                    "Panel layout direction",
                    ["Right to left", "Left to right"],
                    index=0
                )

            direction = "right_to_left" if direction_label == "Right to left" else "left_to_right"

            submitted = st.form_submit_button("Calculate estimate")

        if not submitted:
            st.info("Fill in the measurements and press **Calculate estimate**.")
            st.stop()

        # Convert to mm
        if units.startswith("Meters"):
            A_mm = roof_width * 1000.0
            H_mm = roof_length * 1000.0
            C_mm = bump_start * 1000.0
            W_mm = bump_width * 1000.0
            E_mm = bump_depth * 1000.0
        else:
            A_mm = roof_width
            H_mm = roof_length
            C_mm = bump_start
            W_mm = bump_width
            E_mm = bump_depth

        # Derived
        G_mm = W_mm / 2.0
        coverage_w = max(1.0, raw_w - side_lap)

        # Sanity checks (customer-friendly)
        errors = []
        if A_mm <= 0 or H_mm <= 0:
            errors.append("Roof width and roof length must be greater than 0.")
        if W_mm < 0 or E_mm < 0 or C_mm < 0:
            errors.append("Bump-out values cannot be negative.")
        if C_mm + W_mm > A_mm:
            errors.append("Bump-out start + bump-out width must be within the roof width.")
        if E_mm > H_mm:
            errors.append("Bump-out depth cannot be greater than the roof length.")

        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        # Build polygon
        poly = face_polygon_gable_notch(A_mm, H_mm, C_mm, G_mm, E_mm)

        if not poly.is_valid:
            st.error("These measurements produce an invalid shape. Double-check your numbers.")
            st.stop()

        # Panelize
        panels = panelize_face(poly, coverage_w=coverage_w, direction=direction)

        # Exact area from polygon
        area_m2 = poly.area / 1e6
        area_m2_waste = area_m2 * (1 + waste_pct / 100.0)
        cost = area_m2_waste * price_per_m2

        st.subheader("Estimate summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated roof face area", f"{area_m2:.2f} m²")
        c2.metric("Estimated area incl. waste", f"{area_m2_waste:.2f} m²", delta=f"+{waste_pct:.1f}%")
        c3.metric("Estimated material cost", f"{cost:,.0f} SEK".replace(",", " "))

        st.caption("Note: This is a rough estimate. Final quote may change after verification / site check.")

    with right:
        st.subheader("Preview (for confirmation)")
        fig = plot_face_and_panels(poly, panels)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Installer details (cut list / panel data)"):
            df = pd.DataFrame([{
                "Panel #": p.idx,
                "U start (mm)": round(p.u0, 1),
                "U end (mm)": round(p.u1, 1),
                "Width (mm)": round(p.width, 1),
                "Cut length (mm)": round(p.max_len, 1),
                "Left length (mm)": round(p.left_len, 1),
                "Right length (mm)": round(p.right_len, 1),
                "Note": p.note
            } for p in panels])
            st.dataframe(df, use_container_width=True)

            longest = max((p.max_len for p in panels), default=0)
            st.write(f"**Number of panels:** {len(panels)}")
            st.write(f"**Longest required panel length (approx):** {longest:.0f} mm")
