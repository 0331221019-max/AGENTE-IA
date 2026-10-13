"""
graficos.py
Gráficos Plotly reutilizados por los paneles y por el agente.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import datos as D

COLORES = ["#1E3A5F", "#C9A96E", "#5B8A72", "#B5654A", "#6C7BA8", "#8E5A7A",
           "#3E8A9A", "#A3A15A", "#7A8494", "#D08C4F"]
COLOR_NIVEL = {"A": "#16a34a", "B": "#65a30d", "C": "#eab308",
               "D": "#f97316", "E": "#dc2626", "F": "#7f1d1d"}
NOMBRE_STD = {"_ubic": "Ubicación", "_tipo": "Tipo de vehículo", "_sent": "Sentido"}


def _estilo(fig: go.Figure, alto: int = 380) -> go.Figure:
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2), colorway=COLORES,
                      hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#1B2433"),
                      title_font=dict(family="Playfair Display, serif", size=17, color="#1E3A5F"))
    fig.update_xaxes(gridcolor="#E6E1D6", zerolinecolor="#E6E1D6")
    fig.update_yaxes(gridcolor="#E6E1D6", zerolinecolor="#E6E1D6")
    return fig


def serie_temporal(d: pd.DataFrame, por: str | None = None) -> go.Figure:
    dias = d["_ts"].dt.normalize().nunique()
    if D.tiene_hora(d) and dias <= 4:
        s = D.flujo_horario(d, por).rename(columns={"_h": "Fecha"})
        titulo = "Flujo vehicular por hora"
    else:
        s = D.flujo_diario(d, por).rename(columns={"_d": "Fecha"})
        titulo = "Flujo vehicular diario"
    fig = px.line(s, x="Fecha", y="_cant", color=por, markers=dias < 60,
                  labels={"_cant": "Vehículos", **NOMBRE_STD}, title=titulo)
    return _estilo(fig)


def perfil_horario(d: pd.DataFrame, por: str | None = None) -> go.Figure:
    p = D.perfil_horario(d, por)
    fig = px.line(p, x="hora", y="promedio", color=por, markers=True,
                  labels={"hora": "Hora del día", "promedio": "Vehículos/hora (promedio)", **NOMBRE_STD},
                  title="Perfil horario típico")
    fig.update_xaxes(dtick=1)
    if not por and len(p):
        i = p["promedio"].idxmax()
        fig.add_annotation(x=p.loc[i, "hora"], y=p.loc[i, "promedio"], text="Hora pico",
                           showarrow=True, arrowhead=2)
    return _estilo(fig)


def perfil_semanal(d: pd.DataFrame) -> go.Figure:
    p = D.perfil_semanal(d)
    fig = px.bar(p, x="dia", y="promedio", text_auto=".0f",
                 labels={"dia": "", "promedio": "Vehículos/día (promedio)"},
                 title="Flujo promedio por día de la semana")
    return _estilo(fig)


def mapa_calor(d: pd.DataFrame) -> go.Figure:
    m = D.matriz_dia_hora(d)
    fig = px.imshow(m, aspect="auto", color_continuous_scale="YlOrRd",
                    labels={"x": "Hora", "y": "", "color": "Veh/h"},
                    title="Mapa de calor: día de la semana × hora")
    fig.update_xaxes(dtick=1)
    return _estilo(fig, 360)


def por_categoria(d: pd.DataFrame, std: str, dona: bool = False, top: int = 15) -> go.Figure:
    c = D.por_categoria(d, std, top)
    nombre = NOMBRE_STD.get(std, std)
    if dona:
        fig = px.pie(c, names="categoria", values="vehiculos", hole=0.5,
                     title=f"Composición por {nombre.lower()}")
        fig.update_traces(textinfo="percent+label")
    else:
        fig = px.bar(c.sort_values("vehiculos"), x="vehiculos", y="categoria", orientation="h",
                     text_auto=".3s", labels={"vehiculos": "Vehículos", "categoria": ""},
                     title=f"Vehículos por {nombre.lower()}")
    return _estilo(fig)


def apilado_tipo_hora(d: pd.DataFrame) -> go.Figure:
    fh = D.flujo_horario(d, "_tipo")
    fh["hora"] = fh["_h"].dt.hour
    p = fh.groupby(["hora", "_tipo"], observed=True)["_cant"].mean().reset_index()
    fig = px.bar(p, x="hora", y="_cant", color="_tipo",
                 labels={"hora": "Hora", "_cant": "Vehículos/hora", "_tipo": "Tipo"},
                 title="Composición vehicular por hora")
    fig.update_xaxes(dtick=1)
    return _estilo(fig)


def velocidad_vs_flujo(d: pd.DataFrame) -> go.Figure:
    """Diagrama fundamental simplificado: flujo horario vs velocidad media."""
    x = d.dropna(subset=["_ts"]).copy()
    x["_h"] = x["_ts"].dt.floor("h")
    por = "_ubic" if D.tiene(d, "_ubic") else None
    g = x.groupby(["_h"] + ([por] if por else []), observed=True).agg(
        flujo=("_cant", "sum"), velocidad=("_vel", "mean")).reset_index()
    fig = px.scatter(g, x="flujo", y="velocidad", color=por, opacity=0.6,
                     labels={"flujo": "Flujo (veh/h)", "velocidad": "Velocidad media (km/h)", **NOMBRE_STD},
                     title="Relación flujo – velocidad")
    return _estilo(fig)


def nivel_servicio(d: pd.DataFrame, capacidad: float) -> go.Figure:
    por = "_ubic" if D.tiene(d, "_ubic") else None
    fh = D.flujo_horario(d, por)
    fh["hora"] = fh["_h"].dt.hour
    g = fh.groupby(["hora"] + ([por] if por else []), observed=True)["_cant"].mean().reset_index()
    g["vc"] = g["_cant"] / capacidad
    fig = px.line(g, x="hora", y="vc", color=por, markers=True,
                  labels={"hora": "Hora", "vc": "Relación v/c", **NOMBRE_STD},
                  title="Relación volumen/capacidad (v/c) por hora")
    for lim, letra, _ in D.NIVELES[:-1]:
        fig.add_hline(y=lim, line_dash="dot", line_color=COLOR_NIVEL[letra],
                      annotation_text=f"Límite {letra}", annotation_position="top left",
                      annotation_font_size=10, annotation_font_color=COLOR_NIVEL[letra])
    fig.update_xaxes(dtick=1)
    return _estilo(fig, 420)


def pronostico(res: dict, ultimas: int = 7 * 24) -> go.Figure:
    h, p = res["historia"].tail(ultimas), res["tabla"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["fecha_hora"], y=h["vehiculos"], name="Histórico",
                             mode="lines", line=dict(color=COLORES[0])))
    fig.add_trace(go.Scatter(x=list(p["fecha_hora"]) + list(p["fecha_hora"][::-1]),
                             y=list(p["maximo"]) + list(p["minimo"][::-1]), fill="toself",
                             fillcolor="rgba(201,169,110,0.25)", line=dict(width=0),
                             name="Rango probable (80 %)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=p["fecha_hora"], y=p["pronostico"], name="Pronóstico",
                             mode="lines+markers", line=dict(color=COLORES[1], dash="dash")))
    fig.update_layout(title=f"Pronóstico de flujo vehicular (por {res['granularidad']})",
                      yaxis_title="Vehículos")
    return _estilo(fig, 420)


def desde_tabla(tabla: pd.DataFrame, tipo: str = "barras", titulo: str = "") -> go.Figure:
    """Gráfico genérico a partir del resultado de una consulta del agente."""
    cols = list(tabla.columns)
    y = cols[-1]
    x = cols[0]
    color = cols[1] if len(cols) > 2 else None
    t = tabla.copy()
    t[x] = t[x].astype(str)
    if tipo == "lineas":
        fig = px.line(t, x=x, y=y, color=color, markers=True, title=titulo)
    elif tipo == "pastel":
        fig = px.pie(t, names=x, values=y, hole=0.45, title=titulo)
    else:
        fig = px.bar(t, x=x, y=y, color=color, barmode="group", text_auto=".3s", title=titulo)
    return _estilo(fig)
