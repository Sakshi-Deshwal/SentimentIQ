
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INDIGO = "#4F46E5"
SENT = {"Positive": "#16A34A", "Negative": "#DC2626", "Neutral": "#64748B"}
FAMILY = {"Traditional ML": "#0EA5E9", "Neural network": "#8B5CF6"}
FONT = dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", size=13, color="#0F172A")


def _layout(fig: go.Figure, title: Optional[str] = None, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)) if title else None, height=height,
                      font=FONT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0")
    return fig


def sentiment_distribution(counts: dict, labels, title: str = "Sentiment distribution") -> go.Figure:
    vals = [int(counts.get(l, 0)) for l in labels]
    total = max(sum(vals), 1)
    fig = go.Figure(go.Bar(x=list(labels), y=vals, marker_color=[SENT[l] for l in labels],
                           text=[f"{v:,}<br>{v / total:.1%}" for v in vals], textposition="outside",
                           hovertemplate="%{x}: %{y:,}<extra></extra>"))
    fig.update_yaxes(range=[0, max(vals + [1]) * 1.18], title="Reviews")
    return _layout(fig, title, legend=False)


def domain_sentiment(table: pd.DataFrame, labels, normalize: bool = True, title: str = "Sentiment by domain") -> go.Figure:
    """``table``: index = domain, columns = sentiment labels, values = counts."""
    t = table.reindex(columns=list(labels), fill_value=0)
    data = t.div(t.sum(axis=1).replace(0, 1), axis=0) * 100 if normalize else t
    fig = go.Figure()
    for l in labels:
        fig.add_bar(name=l, y=data.index.tolist(), x=data[l], orientation="h", marker_color=SENT[l],
                    hovertemplate=f"{l}: %{{x:.1f}}{'%' if normalize else ''}<extra>%{{y}}</extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="% of the domain's reviews" if normalize else "Reviews")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, height=120 + 55 * len(data))


def confusion_matrix_fig(cm, labels, normalize: bool = False, title: Optional[str] = None, height: int = 400) -> go.Figure:
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.where(cm.sum(1, keepdims=True) == 0, 1, cm.sum(1, keepdims=True)) if normalize else cm
    text = [[f"{shown[i, j]:.1%}" if normalize else f"{int(cm[i, j]):,}" for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(z=shown, x=list(labels), y=list(labels), text=text, texttemplate="%{text}",
                               colorscale="Blues", showscale=False, zmin=0, zmax=1 if normalize else None,
                               hovertemplate="true %{y} -> predicted %{x}: %{text}<extra></extra>"))
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="True", autorange="reversed")
    return _layout(fig, title, height=height, legend=False)


def metric_comparison(df: pd.DataFrame, metrics, title: str = "Test performance by model") -> go.Figure:
    """``df``: columns Model, Family + metric columns."""
    fig = go.Figure()
    palette = ["#4F46E5", "#14B8A6", "#F59E0B", "#EC4899"]
    for i, m in enumerate(metrics):
        fig.add_bar(name=m, x=df["Model"], y=df[m], marker_color=palette[i % 4],
                    text=[f"{v:.3f}" for v in df[m]], textposition="outside")
    lo = max(0.0, float(df[list(metrics)].min().min()) - 0.08)
    fig.update_yaxes(range=[lo, min(1.02, float(df[list(metrics)].max().max()) + 0.04)])
    fig.update_layout(barmode="group")
    return _layout(fig, title)


def split_gap(df: pd.DataFrame, title: str = "Accuracy by split (train vs validation vs test)") -> go.Figure:
    fig = go.Figure()
    for col, color in (("Train", "#94A3B8"), ("Validation", "#6366F1"), ("Test", "#16A34A")):
        fig.add_bar(name=col, x=df["Model"], y=df[col], marker_color=color, text=[f"{v:.3f}" for v in df[col]],
                    textposition="outside")
    lo = max(0.0, float(df[["Train", "Validation", "Test"]].min().min()) - 0.08)
    fig.update_yaxes(range=[lo, min(1.02, float(df[["Train", "Validation", "Test"]].max().max()) + 0.04)])
    fig.update_layout(barmode="group")
    return _layout(fig, title)


def per_class_bars(report: dict, labels, title: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    for metric, name, color in (("precision", "Precision", "#6366F1"), ("recall", "Recall", "#14B8A6"),
                                ("f1-score", "F1", "#F59E0B")):
        fig.add_bar(name=name, x=list(labels), y=[report[l][metric] for l in labels], marker_color=color,
                    text=[f"{report[l][metric]:.3f}" for l in labels], textposition="outside")
    fig.update_yaxes(range=[0, 1.1])
    fig.update_layout(barmode="group")
    return _layout(fig, title, height=330)


def learning_curves(history: dict, best_epoch: Optional[int] = None, title: str = "") -> go.Figure:
    """Training vs validation loss and accuracy from the per-epoch history recorded during training."""
    ep = history["epoch"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"), horizontal_spacing=0.1)
    for col, (tr, va) in enumerate((("train_loss", "val_loss"), ("train_acc", "val_acc")), start=1):
        fig.add_scatter(x=ep, y=history[tr], name="Training", mode="lines+markers", line=dict(color="#6366F1"),
                        legendgroup="tr", showlegend=col == 1, row=1, col=col)
        fig.add_scatter(x=ep, y=history[va], name="Validation", mode="lines+markers", line=dict(color="#F59E0B"),
                        legendgroup="va", showlegend=col == 1, row=1, col=col)
        if best_epoch:
            fig.add_vline(x=best_epoch, line=dict(color="#94A3B8", dash="dot"), row=1, col=col)
    fig.update_xaxes(title="Epoch", dtick=1 if len(ep) <= 15 else None)
    return _layout(fig, title or None, height=340)


def class_f1_heatmap(df: pd.DataFrame, title: str = "Per-class F1 on the test split") -> go.Figure:
    """``df``: index = model name, columns = class labels, values = F1."""
    fig = go.Figure(go.Heatmap(z=df.values, x=list(df.columns), y=list(df.index), colorscale="Blues",
                               text=[[f"{v:.3f}" for v in row] for row in df.values], texttemplate="%{text}",
                               zmin=max(0, float(np.nanmin(df.values)) - 0.1), zmax=1, showscale=False))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, height=110 + 50 * len(df), legend=False)


def error_bars(df: pd.DataFrame, x: str, y: str = "errors", color: Optional[str] = None, title: str = "") -> go.Figure:
    fig = go.Figure()
    if color:
        for key, g in df.groupby(color):
            fig.add_bar(name=str(key), x=g[x], y=g[y], marker_color=SENT.get(key))
        fig.update_layout(barmode="group")
    else:
        fig.add_bar(x=df[x], y=df[y], marker_color=INDIGO)
    return _layout(fig, title, height=330, legend=bool(color))


def probability_bars(probs: dict, labels, title: str = "Estimated class probability", pct: bool = True) -> go.Figure:
    vals = [float(probs[l]) for l in labels]
    fig = go.Figure(go.Bar(x=vals, y=list(labels), orientation="h", marker_color=[SENT[l] for l in labels],
                           text=[f"{v:.1%}" if pct else f"{v:+.2f}" for v in vals], textposition="outside"))
    if pct:
        fig.update_xaxes(range=[0, 1.12])
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, height=90 + 55 * len(labels), legend=False)


def cross_domain_bars(df: pd.DataFrame, title: str = "Macro F1: cross-domain vs in-domain baseline") -> go.Figure:
    """``df``: columns Experiment, 'Cross-domain macro F1', 'In-domain baseline macro F1'."""
    fig = go.Figure()
    fig.add_bar(name="In-domain baseline (random split)", x=df["Experiment"], y=df["In-domain baseline macro F1"],
                marker_color="#94A3B8")
    fig.add_bar(name="Held-out domain", x=df["Experiment"], y=df["Cross-domain macro F1"], marker_color=INDIGO,
                text=[f"{v:.3f}" for v in df["Cross-domain macro F1"]], textposition="outside")
    fig.update_yaxes(range=[0, 1.05], title="Macro F1")
    fig.update_layout(barmode="group")
    return _layout(fig, title)


def bar_counts(names, values, title: str = "", color: str = INDIGO, ytitle: str = "Reviews") -> go.Figure:
    fig = go.Figure(go.Bar(x=list(names), y=list(values), marker_color=color,
                           text=[f"{int(v):,}" for v in values], textposition="outside"))
    fig.update_yaxes(range=[0, max(list(values) + [1]) * 1.18], title=ytitle)
    return _layout(fig, title, height=330, legend=False)


def sentiment_trend(share: pd.DataFrame, labels, title: str = "Sentiment over time") -> go.Figure:
    """Lines of the share of each sentiment per period (index = period start, columns include ``labels``)."""
    fig = go.Figure()
    for l in labels:
        fig.add_trace(go.Scatter(x=share.index, y=share[l], mode="lines+markers", name=l,
                                 line=dict(color=SENT.get(l, INDIGO), width=2.5),
                                 hovertemplate="%{x|%Y-%m-%d}<br>" + l + ": %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero", title="Share of reviews")
    fig.update_xaxes(title=f"Period ({share.attrs.get('period', '')})" if share.attrs.get("period") else None)
    return _layout(fig, title)
