"""
app.py — Agente IA de Flujo Vehicular (Streamlit + Supabase)
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

import agente as A
import datos as D
import graficos as G

st.set_page_config(page_title="Agente IA · Flujo Vehicular", page_icon="🚦", layout="wide")


def secreto(nombre: str, defecto: str = "") -> str:
    try:
        return st.secrets.get(nombre, defecto)
    except Exception:  # no existe .streamlit/secrets.toml
        return defecto


SUPABASE_URL = secreto("SUPABASE_URL", "https://kpqfmaatyhbhsbxehwzp.supabase.co")
SUPABASE_KEY = secreto("SUPABASE_KEY", "sb_publishable_2c3IJNFfYK9tyGdZ8e84mQ_umS6wMvJ")
TABLA_DEFECTO = secreto("SUPABASE_TABLA", "flujo_vehicular")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem;}
  [data-testid="stMetricValue"] {font-size: 1.55rem; color:#1E3A5F; font-weight:600;}
  [data-testid="stMetric"] {background:#FFFFFF; box-shadow:0 1px 3px rgba(15,30,51,.06);}
  .nivel {display:inline-block;padding:2px 10px;border-radius:12px;color:white;font-weight:600}

  /* ---- Barra lateral ---- */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1E33 0%, #132640 60%, #0B1627 100%);
    border-right: 1px solid rgba(201,169,110,.25);
  }
  .marca {display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1.4rem;}
  .marca .logo {width:42px; height:42px; border-radius:11px; display:grid; place-items:center;
    background:linear-gradient(135deg,#C9A96E,#8E7240); color:#0F1E33; font-size:1.25rem;
    box-shadow:0 4px 14px rgba(201,169,110,.25);}
  .marca .nombre {font-family:'Playfair Display',serif; font-size:1.55rem; font-weight:600;
    color:#F3EEE3; letter-spacing:.3px; line-height:1;}
  .marca .sub {font-size:.68rem; letter-spacing:2.2px; text-transform:uppercase; color:#C9A96E; margin-top:.3rem;}
  .seccion {font-size:.68rem; letter-spacing:2px; text-transform:uppercase; color:#8C9AB0;
    margin:1.3rem 0 .4rem; font-weight:600;}

  /* Navegación tipo menú (radio sin círculos) */
  [data-testid="stSidebar"] [role="radiogroup"] {gap:.2rem;}
  [data-testid="stSidebar"] [role="radiogroup"] > label {
    padding:.55rem .8rem; border-radius:9px; margin:0; width:100%;
    border-left:3px solid transparent; transition:background .15s, border-color .15s;}
  [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {display:none;}
  [data-testid="stSidebar"] [role="radiogroup"] > label:hover {background:rgba(255,255,255,.05);}
  [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background:rgba(201,169,110,.13); border-left-color:#C9A96E;}
  [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {color:#E9D3A6; font-weight:600;}

  [data-testid="stSidebar"] [data-testid="stExpander"] details {
    background:rgba(255,255,255,.03); border-color:rgba(201,169,110,.22);}
  [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover p {color:#E9D3A6;}
  [data-testid="stSidebar"] .stButton button {
    background:transparent; border:1px solid #C9A96E; color:#E9D3A6;}
  [data-testid="stSidebar"] .stButton button:hover {background:#C9A96E; color:#0F1E33;}
  [data-testid="stSidebar"] h3 {font-family:Inter,sans-serif; font-size:.72rem !important;
    letter-spacing:2px; text-transform:uppercase; color:#8C9AB0 !important; font-weight:600;}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Carga de datos (con caché)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Leyendo la tabla de Supabase…")
def cargar(url: str, key: str, tabla: str) -> pd.DataFrame:
    return D.cargar_tabla(url, key, tabla)


@st.cache_data(ttl=600, show_spinner=False)
def tablas_disponibles(url: str, key: str) -> list[str]:
    return D.listar_tablas(url, key)


@st.cache_data(show_spinner=False)
def ejemplo() -> pd.DataFrame:
    return D.datos_ejemplo()


@st.cache_data(show_spinner="Preparando datos…")
def preparar(df: pd.DataFrame, roles_txt: str) -> pd.DataFrame:
    return D.preparar(df, json.loads(roles_txt))


def fmt(x, dec=0) -> str:
    try:
        return f"{x:,.{dec}f}".replace(",", " ")
    except (TypeError, ValueError):
        return str(x)


# ----------------------------------------------------------------------------
# Barra lateral: navegación, conexión y columnas
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""<div class="marca"><div class="logo">🚦</div>
      <div><div class="nombre">MoviAgente</div><div class="sub">Inteligencia vial</div></div></div>
      <div class="seccion">Módulos</div>""", unsafe_allow_html=True)
    pagina = st.radio("Módulo", ["🤖 Agente IA", "📊 Panel general", "🚥 Congestión y nivel de servicio",
                                 "🔮 Pronóstico", "⚠️ Anomalías", "📋 Datos y reporte"],
                      label_visibility="collapsed")

    with st.expander("🔌 Conexión a Supabase", expanded=False):
        url = st.text_input("URL del proyecto", SUPABASE_URL)
        key = st.text_input("API key (publishable)", SUPABASE_KEY, type="password")
        usar_ejemplo = st.toggle("Usar datos de ejemplo", value=False,
                                 help="Datos simulados para probar la app sin conexión.")
        lista = [] if usar_ejemplo else tablas_disponibles(url, key)
        if lista:
            i = lista.index(TABLA_DEFECTO) if TABLA_DEFECTO in lista else 0
            tabla = st.selectbox("Tabla", lista, index=i)
        else:
            tabla = st.text_input("Nombre de la tabla", TABLA_DEFECTO,
                                  help="Tal cual aparece en Supabase (Table Editor).")
        if st.button("🔄 Recargar datos", width="stretch"):
            st.cache_data.clear()
            st.rerun()

# ----------------------------------------------------------------------------
# Lectura de la tabla
# ----------------------------------------------------------------------------
if usar_ejemplo:
    crudo, tabla = ejemplo(), "datos_de_ejemplo"
else:
    try:
        crudo = cargar(url, key, tabla)
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo leer la tabla **{tabla}**.\n\n{e}")
        st.info("Revisa el nombre de la tabla en la barra lateral (🔌 Conexión) o activa "
                "**Usar datos de ejemplo** para probar la app.")
        st.stop()

if crudo.empty:
    st.warning(f"La tabla **{tabla}** respondió sin filas.")
    st.markdown("""Si en Supabase sí tiene datos, lo más probable es que **RLS (Row Level Security)** esté
activado sin una política de lectura. Ejecuta esto en el *SQL Editor* de Supabase (cambia el nombre de la tabla):""")
    st.code(f'create policy "lectura publica" on public."{tabla}"\nfor select to anon using (true);', language="sql")
    st.stop()

# ----------------------------------------------------------------------------
# Detección / ajuste de columnas
# ----------------------------------------------------------------------------
clave_roles = f"roles::{tabla}::{','.join(map(str, crudo.columns))}"
if clave_roles not in st.session_state:
    st.session_state[clave_roles] = D.detectar_columnas(crudo)
roles = st.session_state[clave_roles]

with st.sidebar:
    with st.expander("🧭 Columnas detectadas", expanded=False):
        st.caption("Se detectan solas. Corrígelas si alguna no es la correcta.")
        opciones = ["(ninguna)"] + list(crudo.columns)
        nuevos = {}
        for rol in D.ROLES:
            if rol == "tipo" and roles.get("columnas_tipo"):
                continue
            actual = roles.get(rol)
            nuevos[rol] = st.selectbox(D.NOMBRE_ROL[rol], opciones,
                                       index=opciones.index(actual) if actual in opciones else 0,
                                       key=f"rol_{rol}_{tabla}")
        nuevos["columnas_tipo"] = st.multiselect(
            "Columnas por tipo de vehículo (formato ancho)",
            list(crudo.columns), default=roles.get("columnas_tipo") or [],
            help="Úsalo si la tabla tiene una columna por tipo: autos, buses, motos…")
        nuevos = {k: (None if v == "(ninguna)" else v) for k, v in nuevos.items()}
        nuevos.setdefault("tipo", None)
        if nuevos != roles:
            st.session_state[clave_roles] = roles = nuevos

d_total = preparar(crudo, json.dumps(roles, default=str))
TIENE_TS, TIENE_HORA = D.tiene(d_total, "_ts"), D.tiene_hora(d_total)

# ----------------------------------------------------------------------------
# Filtros globales
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("🎛️ Filtros")
    desde = hasta = None
    if TIENE_TS:
        fmin, fmax = d_total["_ts"].min().date(), d_total["_ts"].max().date()
        rango = st.date_input("Rango de fechas", (fmin, fmax), min_value=fmin, max_value=fmax,
                              format="DD/MM/YYYY")
        if isinstance(rango, (list, tuple)) and len(rango) == 2:
            desde, hasta = rango
    sel = {}
    for std, etiqueta in (("_ubic", "Ubicaciones"), ("_tipo", "Tipos de vehículo"), ("_sent", "Sentidos")):
        if D.tiene(d_total, std):
            sel[std] = st.multiselect(etiqueta, sorted(d_total[std].dropna().unique()), placeholder="Todos")
    h_ini, h_fin = (0, 23)
    if TIENE_HORA:
        h_ini, h_fin = st.slider("Horas del día", 0, 23, (0, 23))

    st.subheader("🧠 Inteligencia artificial")
    proveedor = st.selectbox("Proveedor", list(A.PROVEEDORES))
    info_p = A.PROVEEDORES[proveedor]
    api_key = st.text_input("API key", secreto(info_p["secreto"]), key=f"key_{proveedor}",
                            type="password", help=f"Consíguela en {info_p['clave']}. Sin clave funciona el modo local.")
    modelo = st.text_input("Modelo", info_p["modelo"], key=f"modelo_{proveedor}")
    capacidad = st.number_input("Capacidad de la vía (veh/h)", 100, 20000, 1800, 100,
                                help="Para la relación v/c. Referencia: ~1 800 veh/h por carril en vía urbana.")

d = D.filtrar(d_total, desde, hasta, sel.get("_ubic"), sel.get("_tipo"), sel.get("_sent"),
              h_ini if TIENE_HORA and (h_ini, h_fin) != (0, 23) else None,
              h_fin if TIENE_HORA and (h_ini, h_fin) != (0, 23) else None)

partes_filtro = []
if desde and (desde, hasta) != (d_total["_ts"].min().date(), d_total["_ts"].max().date()):
    partes_filtro.append(f"fechas {desde:%d/%m/%Y}–{hasta:%d/%m/%Y}")
for std, n in (("_ubic", "ubicaciones"), ("_tipo", "tipos"), ("_sent", "sentidos")):
    if sel.get(std):
        partes_filtro.append(f"{n}: {', '.join(sel[std])}")
if TIENE_HORA and (h_ini, h_fin) != (0, 23):
    partes_filtro.append(f"horas {h_ini}–{h_fin}")
FILTROS_TXT = "; ".join(partes_filtro) or "ninguno"

# Encabezado común
st.markdown(f"### {pagina}")
st.caption(f"Tabla **{tabla}** · {fmt(len(crudo))} filas · filtros: {FILTROS_TXT}"
           + ("" if TIENE_TS else " · ⚠️ sin columna de fecha detectada"))
if d.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()


# ============================================================================
# 🤖 AGENTE IA
# ============================================================================
def pagina_agente():
    if "chat" not in st.session_state:
        st.session_state.chat = []
    modo = f"IA: {proveedor.split(' –')[0]} · {modelo}" if api_key else "Modo local (sin API key)"
    c1, c2 = st.columns([4, 1])
    c1.info(f"**{modo}**. Pregúntame por horas pico, congestión, tipos de vehículo, "
            "pronósticos, anomalías o pídeme gráficos y recomendaciones.", icon="🤖")
    if c2.button("🧹 Nueva conversación", width="stretch"):
        st.session_state.chat = []
        st.session_state.pop("pill_sug", None)
        st.rerun()

    sugerencias = ["¿Cuáles son las horas pico?", "¿Qué ubicación está más congestionada?",
                   "Pronostica el flujo de las próximas 24 horas", "¿Hubo incidentes o anomalías?",
                   "¿Cómo es la composición por tipo de vehículo?",
                   "Dame un resumen y recomienda medidas para reducir la congestión"]
    if not st.session_state.chat:
        elegida = st.pills("Prueba con:", sugerencias, key="pill_sug")
        if elegida:
            st.session_state.pendiente = elegida

    for i, m in enumerate(st.session_state.chat):
        with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🚦"):
            st.markdown(m["content"])
            for j, fig in enumerate(m.get("graficos", [])):
                st.plotly_chart(fig, key=f"g_{i}_{j}")
            if m.get("pasos"):
                with st.expander(f"🔧 Herramientas usadas ({len(m['pasos'])})"):
                    for p in m["pasos"]:
                        st.code(f"{p['herramienta']}({json.dumps(p['argumentos'], ensure_ascii=False)})",
                                language="python")

    pregunta = st.chat_input("Escribe tu pregunta sobre el flujo vehicular…")
    pregunta = pregunta or st.session_state.pop("pendiente", None)
    if not pregunta:
        return
    st.session_state.chat.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(pregunta)
    with st.chat_message("assistant", avatar="🚦"):
        with st.spinner("Analizando los datos…"):
            try:
                r = A.responder(pregunta, st.session_state.chat[:-1], d,
                                {"proveedor": proveedor, "modelo": modelo, "api_key": api_key,
                                 "capacidad": capacidad}, tabla, FILTROS_TXT)
            except Exception as e:  # noqa: BLE001
                txt = str(e)
                pista = ("Revisa que la API key sea válida." if any(x in txt.lower() for x in ("401", "auth", "api key", "api_key"))
                         else "Revisa el nombre del modelo." if "404" in txt or "model" in txt.lower()
                         else "Intenta de nuevo en unos segundos." if "429" in txt or "rate" in txt.lower() else "")
                r = {"texto": f"⚠️ No pude consultar al modelo de IA: `{txt[:400]}`\n\n{pista}\n\n"
                              "Mientras tanto, esto es lo que encuentro en modo local:\n\n"
                              + A.agente_local(pregunta, A.Herramientas(d, capacidad)),
                     "graficos": [], "pasos": []}
        st.markdown(r["texto"])
        for j, fig in enumerate(r["graficos"]):
            st.plotly_chart(fig, key=f"g_nuevo_{j}")
    st.session_state.chat.append({"role": "assistant", "content": r["texto"],
                                  "graficos": r["graficos"], "pasos": r["pasos"]})
    st.rerun()


# ============================================================================
# 📊 PANEL GENERAL
# ============================================================================
def pagina_panel():
    k = D.kpis(d)
    tarjetas = [("Vehículos totales", fmt(k["total_vehiculos"]), None)]
    if "promedio_diario" in k:
        tarjetas.append(("Promedio diario", fmt(k["promedio_diario"]), f"{k['dias_con_datos']} días con datos"))
    if "hora_pico" in k:
        tarjetas.append(("Hora pico", k["hora_pico"], f"~{fmt(k['flujo_hora_pico_promedio'])} veh/h en promedio"))
    if "dia_semana_pico" in k:
        tarjetas.append(("Día más cargado", k["dia_semana_pico"], None))
    if "ubicacion_mas_transitada" in k:
        tarjetas.append(("Punto más transitado", k["ubicacion_mas_transitada"], k["ubicacion_mas_transitada"]))
    if "tipo_predominante" in k:
        tarjetas.append(("Tipo predominante", k["tipo_predominante"], None))
    if "velocidad_promedio" in k:
        tarjetas.append(("Velocidad media", f"{fmt(k['velocidad_promedio'], 1)} km/h", None))
    for fila in range(0, len(tarjetas), 4):
        cols = st.columns(4)
        for c, (t, v, ayuda) in zip(cols, tarjetas[fila:fila + 4]):
            c.metric(t, v, help=ayuda, border=True)

    if not TIENE_TS:
        st.info("No se detectó una columna de fecha; se muestran solo los gráficos por categoría. "
                "Puedes indicarla en **🧭 Columnas detectadas**.")
    figs = []
    if TIENE_TS:
        por = st.segmented_control("Separar la serie por", ["Nada", "Ubicación", "Tipo"], default="Nada")
        figs.append(G.serie_temporal(d, {"Ubicación": "_ubic", "Tipo": "_tipo"}.get(por) if por else None))
        if TIENE_HORA:
            figs.append(G.perfil_horario(d))
        figs.append(G.perfil_semanal(d))
        if TIENE_HORA:
            figs.append(G.mapa_calor(d))
    if D.tiene(d, "_tipo"):
        figs.append(G.por_categoria(d, "_tipo", dona=True))
    if D.tiene(d, "_ubic"):
        figs.append(G.por_categoria(d, "_ubic"))
    if TIENE_HORA and D.tiene(d, "_tipo"):
        figs.append(G.apilado_tipo_hora(d))
    if D.tiene(d, "_vel") and TIENE_TS:
        figs.append(G.velocidad_vs_flujo(d))
    if figs:
        st.plotly_chart(figs[0], key="p0")
    for i in range(1, len(figs), 2):
        cols = st.columns(2)
        for j, c in enumerate(cols):
            if i + j < len(figs):
                c.plotly_chart(figs[i + j], key=f"p{i + j}")


# ============================================================================
# 🚥 CONGESTIÓN Y NIVEL DE SERVICIO
# ============================================================================
def pagina_congestion():
    if not TIENE_HORA:
        st.info("Este módulo necesita datos con hora para calcular el flujo horario.")
        return
    st.markdown(f"Capacidad usada: **{fmt(capacidad)} veh/h** (cámbiala en la barra lateral). "
                "El nivel de servicio se estima con la relación volumen/capacidad (v/c), como referencia "
                "simplificada del *Highway Capacity Manual*.")
    t = D.nivel_servicio(d, capacidad)
    colores = G.COLOR_NIVEL
    cols = st.columns(min(len(t), 4) or 1)
    for c, (_, f) in zip(cols, t.head(4).iterrows()):
        with c.container(border=True):
            st.markdown(f"**{f['ubicacion']}**")
            st.markdown(f"<span class='nivel' style='background:{colores[f['nivel_peor']]}'>Peor: {f['nivel_peor']}</span> "
                        f"<span class='nivel' style='background:{colores[f['nivel_tipico']]}'>Típico: {f['nivel_tipico']}</span>",
                        unsafe_allow_html=True)
            st.caption(f"v/c máx. {f['vc_maximo']} · {f['pct_horas_E_F']} % de horas en E/F")
    st.plotly_chart(G.nivel_servicio(d, capacidad), key="ns")
    st.dataframe(t, hide_index=True, column_config={
        "ubicacion": "Ubicación", "flujo_hora_promedio": "Flujo prom. (veh/h)",
        "flujo_hora_maximo": "Flujo máx. (veh/h)", "vc_promedio": "v/c prom.",
        "vc_maximo": st.column_config.ProgressColumn("v/c máx.", min_value=0, max_value=1.2, format="%.2f"),
        "nivel_tipico": "Nivel típico", "nivel_peor": "Nivel peor", "pct_horas_E_F": "% horas E/F",
        "hora_mas_critica": "Hora más crítica"})
    with st.expander("¿Qué significa cada nivel?"):
        st.table(pd.DataFrame([{"Nivel": l, "v/c hasta": "> 1.00" if lim == float("inf") else f"{lim:.2f}",
                                "Descripción": desc} for lim, l, desc in D.NIVELES]))
    criticas = t[t["nivel_peor"].isin(["E", "F"])]
    if len(criticas):
        st.subheader("Recomendaciones")
        hp = D.horas_pico(d, 3)["horas_del_dia_mas_cargadas"]
        horas = ", ".join(f"{f['hora']:02d}:00" for f in hp)
        st.markdown(
            f"- **{', '.join(map(str, criticas['ubicacion']))}** llegan a nivel E/F: revisar el ciclo y reparto de verdes "
            f"del semáforo en las horas críticas ({horas}).\n"
            "- Evaluar restricción de carga pesada y de maniobras de carga/descarga en hora punta.\n"
            "- Controlar paraderos informales y giros a la izquierda conflictivos cerca de esos puntos.\n"
            "- Si la congestión es recurrente, plantear escalonamiento de horarios con empresas y colegios de la zona.")
    else:
        st.success("Ninguna ubicación supera v/c = 0.90 con la capacidad indicada.")


# ============================================================================
# 🔮 PRONÓSTICO
# ============================================================================
def pagina_pronostico():
    if not TIENE_TS:
        st.info("Se necesita una columna de fecha para pronosticar.")
        return
    c1, c2 = st.columns([2, 1])
    unidad = "horas" if TIENE_HORA else "días"
    horizonte = c1.slider(f"Horizonte ({unidad})", 6 if TIENE_HORA else 3, 168 if TIENE_HORA else 60,
                          24 if TIENE_HORA else 14)
    sub = d
    if D.tiene(d, "_ubic"):
        u = c2.selectbox("Ubicación", ["Todas"] + sorted(d["_ubic"].dropna().unique()))
        if u != "Todas":
            sub = d[d["_ubic"] == u]
    r = D.pronostico(sub, horizonte)
    if "error" in r:
        st.warning(r["error"])
        return
    m1, m2, m3 = st.columns(3)
    m1.metric("Error de validación (MAPE)", f"{r['mape_validacion_pct']} %" if r["mape_validacion_pct"] is not None else "—",
              help="Se entrena con el 80 % de la historia y se compara contra el 20 % final.", border=True)
    m2.metric("Error absoluto medio", fmt(r["mae_validacion"]) + " veh" if r["mae_validacion"] is not None else "—", border=True)
    m3.metric("Tendencia", f"{r['tendencia_diaria_pct']:+.2f} % por día", border=True)
    st.plotly_chart(G.pronostico(r), key="pron")
    st.caption(f"Método: {r['metodo']}. El rango sombreado cubre ~80 % de los casos esperados.")
    tabla = r["tabla"].round(0)
    st.dataframe(tabla, hide_index=True, column_config={
        "fecha_hora": st.column_config.DatetimeColumn("Fecha y hora", format="DD/MM/YYYY HH:mm"),
        "pronostico": "Pronóstico (veh)", "minimo": "Mínimo probable", "maximo": "Máximo probable"})
    st.download_button("⬇️ Descargar pronóstico (CSV)", tabla.to_csv(index=False).encode("utf-8-sig"),
                       "pronostico_flujo.csv", "text/csv")


# ============================================================================
# ⚠️ ANOMALÍAS
# ============================================================================
def pagina_anomalias():
    if not TIENE_TS:
        st.info("Se necesita una columna de fecha para detectar anomalías.")
        return
    umbral = st.slider("Sensibilidad (z): menor valor = más alertas", 2.0, 6.0, 3.0, 0.5)
    a = D.anomalias(d, umbral)
    st.metric("Periodos atípicos", len(a), border=True)
    if a.empty:
        st.success("No se detectaron periodos atípicos con esta sensibilidad.")
        return
    por = "_ubic" if D.tiene(d, "_ubic") else None
    s = D.flujo_horario(d, por) if TIENE_HORA else D.flujo_diario(d, por)
    t = "_h" if TIENE_HORA else "_d"
    fig = px.line(s, x=t, y="_cant", color=por, labels={t: "Fecha", "_cant": "Vehículos", "_ubic": "Ubicación"},
                  title="Flujo con anomalías marcadas")
    a2 = a.copy()
    a2["momento"] = pd.to_datetime(a2["momento"])
    fig.add_scatter(x=a2["momento"], y=a2["vehiculos"], mode="markers", name="Anomalía",
                    marker=dict(size=11, color="#dc2626", symbol="x"))
    st.plotly_chart(G._estilo(fig, 420), key="anom")
    st.dataframe(a, hide_index=True, column_config={"z": st.column_config.NumberColumn("z", format="%.1f")})


# ============================================================================
# 📋 DATOS Y REPORTE
# ============================================================================
def pagina_datos():
    originales = [c for c in d.columns if not c.startswith("_")]
    st.dataframe(d[originales], hide_index=True, height=420)
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ CSV (filtrado)", d[originales].to_csv(index=False).encode("utf-8-sig"),
                       f"{tabla}.csv", "text/csv", width="stretch")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        d[originales].to_excel(w, sheet_name="datos", index=False)
        if D.tiene(d, "_tipo"):
            D.por_categoria(d, "_tipo").to_excel(w, sheet_name="por_tipo", index=False)
        if D.tiene(d, "_ubic"):
            D.por_categoria(d, "_ubic").to_excel(w, sheet_name="por_ubicacion", index=False)
        if TIENE_HORA:
            D.perfil_horario(d).to_excel(w, sheet_name="perfil_horario", index=False)
    c2.download_button("⬇️ Excel con resúmenes", buf.getvalue(), f"{tabla}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    informe = A.informe_markdown(d, tabla, capacidad)
    c3.download_button("📝 Informe (Markdown)", informe.encode("utf-8"), f"informe_{tabla}_{date.today()}.md",
                       "text/markdown", width="stretch")
    with st.expander("Vista previa del informe", expanded=False):
        st.markdown(informe)
    with st.expander("Columnas detectadas y tipos"):
        st.json({k: v for k, v in roles.items()})


{"🤖 Agente IA": pagina_agente, "📊 Panel general": pagina_panel,
 "🚥 Congestión y nivel de servicio": pagina_congestion, "🔮 Pronóstico": pagina_pronostico,
 "⚠️ Anomalías": pagina_anomalias, "📋 Datos y reporte": pagina_datos}[pagina]()
