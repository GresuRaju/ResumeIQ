"""
visualizer.py — All Plotly charts for ResumeIQ
Dark theme matching the app's aesthetic.
"""

import plotly.graph_objects as go
import plotly.express as px

# ── Theme ─────────────────────────────────────────────────────────────────────
BG        = "#0a0c10"
SURFACE   = "#111318"
CARD      = "#13161d"
BORDER    = "#1e2229"
ACCENT    = "#00e5a0"
ACCENT2   = "#7c6af7"
ACCENT3   = "#f7c948"
DANGER    = "#f05d5e"
MUTED     = "#6b7280"
TEXT      = "#e8eaf0"

LAYOUT_BASE = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="DM Mono, monospace", color=TEXT, size=11),
    margin=dict(l=20, r=20, t=40, b=20),
)


def score_color(score: float) -> str:
    if score >= 75:
        return ACCENT
    elif score >= 50:
        return ACCENT3
    return DANGER


def plot_match_scores(results: list) -> go.Figure:
    names  = [r["name"].split("_")[0][:20] for r in results]
    scores = [r["match_score"] for r in results]
    colors = [score_color(s) for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s}%" for s in scores],
        textposition="outside",
        textfont=dict(color=TEXT, size=11),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Match Score Ranking", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(range=[0, 115], showgrid=True, gridcolor=BORDER,
                   ticksuffix="%", color=MUTED, zeroline=False),
        yaxis=dict(autorange="reversed", showgrid=False, color=MUTED),
        height=max(200, len(results) * 52 + 60),
        bargap=0.35,
    )
    return fig


def plot_ats_scores(results: list) -> go.Figure:
    names = [r["name"].split("_")[0][:20] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Match Score", x=names,
        y=[r["match_score"] for r in results],
        marker_color=ACCENT,
        text=[f"{r['match_score']}%" for r in results],
        textposition="outside", textfont=dict(color=TEXT, size=10),
    ))
    fig.add_trace(go.Bar(
        name="ATS Score", x=names,
        y=[r["ats_score"] for r in results],
        marker_color=ACCENT2,
        text=[f"{r['ats_score']}%" for r in results],
        textposition="outside", textfont=dict(color=TEXT, size=10),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Match Score vs ATS Score", font=dict(color=TEXT, size=13), x=0),
        barmode="group",
        xaxis=dict(showgrid=False, color=MUTED),
        yaxis=dict(range=[0, 115], showgrid=True, gridcolor=BORDER,
                   ticksuffix="%", color=MUTED, zeroline=False),
        legend=dict(font=dict(color=TEXT, size=10), bgcolor=SURFACE, bordercolor=BORDER),
        height=340, bargap=0.25, bargroupgap=0.05,
    )
    return fig


def plot_similarity_bars(results: list) -> go.Figure:
    names = [r["name"].split("_")[0][:20] for r in results]
    sims  = [r["sim"] for r in results]

    fig = go.Figure(go.Bar(
        x=sims, y=names, orientation="h",
        marker=dict(color=sims, colorscale=[[0, ACCENT2], [1, ACCENT]], line=dict(width=0)),
        text=[f"{s}%" for s in sims],
        textposition="outside", textfont=dict(color=TEXT, size=11),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="TF-IDF Cosine Similarity (×100)", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(range=[0, max(sims or [1]) * 1.25 + 5],
                   showgrid=True, gridcolor=BORDER, ticksuffix="%", color=MUTED, zeroline=False),
        yaxis=dict(autorange="reversed", showgrid=False, color=MUTED),
        height=max(200, len(results) * 52 + 60),
        bargap=0.35,
    )
    return fig


def plot_confusion_matrix(cm: dict) -> go.Figure:
    z = [[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]]
    text = [[f"TP\n{cm['tp']}", f"FN\n{cm['fn']}"], [f"FP\n{cm['fp']}", f"TN\n{cm['tn']}"]]
    hover = [["Correctly shortlisted", "Missed candidates"], ["False shortlisting", "Correctly rejected"]]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted Positive", "Predicted Negative"],
        y=["Actual Positive", "Actual Negative"],
        text=text, texttemplate="%{text}",
        textfont=dict(size=14, color=TEXT, family="Syne, sans-serif"),
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        colorscale=[[0.0, "rgb(19, 22, 29)"], [0.5, "rgba(0, 229, 160, 0.18)"], [1.0, "rgba(0, 229, 160, 0.38)"]],
        showscale=False,
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Confusion Matrix — Test Set (426 samples)", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(color=MUTED, side="bottom"),
        yaxis=dict(color=MUTED, autorange="reversed"),
        height=300,
    )
    return fig


def plot_feature_importance() -> go.Figure:
    features = ["Education level", "Role title match", "Experience years", "TF-IDF cosine sim.", "Skill keyword overlap"]
    values   = [0.091, 0.134, 0.189, 0.274, 0.312]
    colors   = ["#38bdf8", DANGER, ACCENT3, ACCENT2, ACCENT]

    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in values],
        textposition="outside", textfont=dict(color=TEXT, size=11),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Feature Importance (Logistic Regression Coefficients)", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(range=[0, 0.38], showgrid=True, gridcolor=BORDER, color=MUTED, zeroline=False),
        yaxis=dict(showgrid=False, color=MUTED),
        height=300, bargap=0.4,
    )
    return fig


def plot_ats_gauge(score: int) -> go.Figure:
    color = score_color(score)
    fig = go.Figure(go.Pie(
        values=[score, 100 - score], hole=0.72,
        marker=dict(colors=[color, BORDER]),
        textinfo="none", hoverinfo="none", sort=False,
    ))
    fig.add_annotation(text=f"<b>{score}</b>", x=0.5, y=0.55,
                       font=dict(size=22, color=color, family="Syne, sans-serif"), showarrow=False)
    fig.add_annotation(text="ATS", x=0.5, y=0.38,
                       font=dict(size=10, color=MUTED), showarrow=False)
    layout = {**LAYOUT_BASE, "showlegend": False, "height": 160}
    layout["margin"] = dict(l=10, r=10, t=10, b=10)
    fig.update_layout(**layout)
    return fig


def plot_candidate_comparison(candidates: list, jd_skills: list) -> go.Figure:
    """Radar chart comparing up to 3 candidates across key dimensions."""
    categories = ["Match Score", "ATS Score", "Skill Overlap", "TF-IDF Sim", "Experience"]
    colors = [ACCENT, ACCENT2, ACCENT3]

    fig = go.Figure()
    for i, r in enumerate(candidates[:3]):
        exp_norm = min(r.get("exp_years", 0) / 5.0 * 100, 100)
        values = [
            r["match_score"],
            r["ats_score"],
            r["skill_score"],
            r["sim"],
            exp_norm,
        ]
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=values_closed, theta=cats_closed,
            fill="toself", name=r["name"][:20],
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=("rgba({},{},{},0.08)".format(
                int(colors[i % len(colors)][1:3], 16),
                int(colors[i % len(colors)][3:5], 16),
                int(colors[i % len(colors)][5:7], 16)
            ) if colors[i % len(colors)].startswith("#") else colors[i % len(colors)]),
            opacity=0.85,
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        polar=dict(
            bgcolor=SURFACE,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER,
                            color=MUTED, tickfont=dict(size=9, color=MUTED)),
            angularaxis=dict(gridcolor=BORDER, color=MUTED,
                             tickfont=dict(size=10, color=TEXT)),
        ),
        showlegend=True,
        legend=dict(font=dict(color=TEXT, size=10), bgcolor=SURFACE, bordercolor=BORDER),
        title=dict(text="Candidate Comparison (Radar)", font=dict(color=TEXT, size=13), x=0),
        height=420,
    )
    return fig


def plot_multi_jd_heatmap(multi_results: dict) -> go.Figure:
    """
    Heatmap: candidates (rows) × job roles (cols) = match score.
    multi_results: {candidate_name: {jd_title: score_dict}}
    """
    candidates = list(multi_results.keys())
    jd_titles  = list(next(iter(multi_results.values())).keys())
    z = [
        [multi_results[c][jd]["match_score"] for jd in jd_titles]
        for c in candidates
    ]
    text = [[f"{v}%" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=jd_titles, y=candidates,
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, color=TEXT),
        colorscale=[[0, DANGER], [0.5, ACCENT3], [1, ACCENT]],
        zmin=0, zmax=100,
        showscale=True,
        colorbar=dict(tickfont=dict(color=TEXT), outlinewidth=0, bgcolor=SURFACE),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Multi-JD Match Score Heatmap", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(color=MUTED, side="bottom"),
        yaxis=dict(color=MUTED, autorange="reversed"),
        height=max(300, len(candidates) * 48 + 80),
    )
    return fig


def plot_gap_bar(gap: dict) -> go.Figure:
    """Bar chart showing matched vs missing skills count."""
    matched_n = len(gap["matched_skills"])
    missing_n = len(gap["missing_skills"])

    fig = go.Figure(go.Bar(
        x=["Matched Skills", "Missing Skills"],
        y=[matched_n, missing_n],
        marker_color=[ACCENT, DANGER],
        text=[matched_n, missing_n],
        textposition="outside",
        textfont=dict(color=TEXT, size=13),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Skills Gap Overview", font=dict(color=TEXT, size=13), x=0),
        xaxis=dict(showgrid=False, color=MUTED),
        yaxis=dict(showgrid=True, gridcolor=BORDER, color=MUTED, zeroline=False),
        height=280, bargap=0.5,
    )
    return fig
