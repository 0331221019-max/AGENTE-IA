"""
agente.py
Agente IA de flujo vehicular: un modelo de lenguaje que decide qué herramientas
de análisis usar (tool calling) sobre los datos de Supabase.
Proveedores: Claude (Anthropic), Gemini (Google) y Groq. Sin API key funciona
un "modo local" basado en reglas.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date

import numpy as np
import pandas as pd

import datos as D
import graficos as G

PROVEEDORES = {
    "Claude (Anthropic)": {"tipo": "anthropic", "modelo": "claude-sonnet-5-5", "base_url": None,
                           "clave": "https://console.anthropic.com/", "secreto": "ANTHROPIC_API_KEY"},
    "Gemini (Google) – tiene capa gratuita": {"tipo": "openai", "modelo": "gemini-3.8-flash",
                                             "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                                             "clave": "https://aistudio.google.com/apikey", "secreto": "GEMINI_API_KEY"},
    "Groq – tiene capa gratuita": {"tipo": "openai", "modelo": "openai/gpt-oss-120b",
                                   "base_url": "https://api.groq.com/openai/v1",
                                   "clave": "https://console.groq.com/keys", "secreto": "GROQ_API_KEY"},
}

# ----------------------------------------------------------------------------
# Definición de herramientas (lo que el modelo puede "llamar")
# ----------------------------------------------------------------------------
_FILTROS = {
    "type": "object",
    "description": "Filtros opcionales. Usa los nombres exactos que devuelve 'valores_disponibles'.",
    "properties": {
        "ubicaciones": {"type": "array", "items": {"type": "string"}},
        "tipos": {"type": "array", "items": {"type": "string"}, "description": "Tipos de vehículo"},
        "sentidos": {"type": "array", "items": {"type": "string"}},
        "dias_semana": {"type": "array", "items": {"type": "string"}, "description": "Lunes..Domingo"},
        "fecha_desde": {"type": "string", "description": "AAAA-MM-DD"},
        "fecha_hasta": {"type": "string", "description": "AAAA-MM-DD"},
        "hora_desde": {"type": "integer", "description": "0-23"},
        "hora_hasta": {"type": "integer", "description": "0-23 (inclusive)"},
    },
}

HERRAMIENTAS = [
    {"name": "valores_disponibles",
     "description": "Lista las ubicaciones, tipos de vehículo, sentidos, rango de fechas y columnas originales. Úsala antes de filtrar si no conoces los nombres exactos.",
     "input_schema": {"type": "object", "properties": {"incluir_columnas": {"type": "boolean"}}}},
    {"name": "resumen_general",
     "description": "Indicadores clave: total de vehículos, promedio diario, hora pico, día pico, ubicación más transitada, tipo predominante y velocidad media.",
     "input_schema": {"type": "object", "properties": {"filtros": _FILTROS}}},
    {"name": "consultar",
     "description": "Consulta flexible: agrupa y agrega el flujo. Ej.: vehículos por hora y ubicación, velocidad promedio por tipo, total por día de la semana.",
     "input_schema": {"type": "object", "properties": {
         "agrupar_por": {"type": "array", "items": {"type": "string"},
                         "description": "Dimensiones: hora, dia_semana, fecha, semana, mes, tipo, ubicacion, sentido, o el nombre de una columna original."},
         "metrica": {"type": "string", "enum": ["vehiculos", "velocidad", "registros"]},
         "agregacion": {"type": "string", "enum": ["suma", "promedio", "maximo", "minimo", "conteo"]},
         "orden": {"type": "string", "enum": ["desc", "asc", "natural"],
                   "description": "desc/asc por el valor; natural por las dimensiones (útil para horas o fechas)."},
         "limite": {"type": "integer"},
         "filtros": _FILTROS}}},
    {"name": "horas_pico",
     "description": "Horas del día con más flujo (promedio) y los momentos puntuales de mayor flujo.",
     "input_schema": {"type": "object", "properties": {
         "n": {"type": "integer"}, "por_ubicacion": {"type": "boolean"}, "filtros": _FILTROS}}},
    {"name": "nivel_de_servicio",
     "description": "Calcula la relación volumen/capacidad (v/c) por hora y un nivel de servicio referencial A–F por ubicación.",
     "input_schema": {"type": "object", "properties": {
         "capacidad_veh_h": {"type": "number", "description": "Capacidad de la vía en vehículos/hora. Si no se indica, se usa la configurada."},
         "filtros": _FILTROS}}},
    {"name": "detectar_anomalias",
     "description": "Detecta horas o días con flujo inusualmente alto o bajo respecto a lo normal (posibles incidentes, cierres o eventos).",
     "input_schema": {"type": "object", "properties": {
         "umbral_z": {"type": "number", "description": "Sensibilidad; 3 por defecto (menor = más alertas)."},
         "filtros": _FILTROS}}},
    {"name": "pronosticar",
     "description": "Pronostica el flujo de las próximas horas (o días si no hay hora) con su rango probable y el error de validación.",
     "input_schema": {"type": "object", "properties": {
         "horizonte": {"type": "integer", "description": "Número de periodos a pronosticar (24 por defecto)."},
         "filtros": _FILTROS}}},
    {"name": "crear_grafico",
     "description": "Muestra un gráfico al usuario. Úsalo cuando pida ver, graficar o comparar algo visualmente.",
     "input_schema": {"type": "object", "properties": {
         "grafico": {"type": "string", "enum": ["serie_temporal", "perfil_horario", "perfil_semanal",
                                                "mapa_calor", "por_tipo", "por_ubicacion",
                                                "composicion_por_hora", "velocidad_vs_flujo",
                                                "nivel_servicio", "pronostico", "consulta"]},
         "separar_por": {"type": "string", "enum": ["ninguno", "ubicacion", "tipo", "sentido"],
                         "description": "Para serie_temporal o perfil_horario: una línea por categoría."},
         "agrupar_por": {"type": "array", "items": {"type": "string"}, "description": "Solo para grafico='consulta'."},
         "metrica": {"type": "string", "enum": ["vehiculos", "velocidad", "registros"]},
         "agregacion": {"type": "string", "enum": ["suma", "promedio", "maximo", "minimo"]},
         "estilo": {"type": "string", "enum": ["barras", "lineas", "pastel"]},
         "titulo": {"type": "string"},
         "filtros": _FILTROS}}},
]

_STD = {"ubicacion": "_ubic", "tipo": "_tipo", "sentido": "_sent"}


# ----------------------------------------------------------------------------
# Ejecución de herramientas
# ----------------------------------------------------------------------------
def _sin_tildes(t: str) -> str:
    return unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower().strip()


def _emparejar(pedidos, disponibles) -> list[str]:
    """Empareja nombres aproximados (sin tildes, mayúsculas ni texto exacto)."""
    disp = [str(x) for x in disponibles]
    out = []
    for p in pedidos or []:
        q = D.normalizar(p)
        exactos = [x for x in disp if D.normalizar(x) == q]
        parciales = [x for x in disp if q and (q in D.normalizar(x) or D.normalizar(x) in q)]
        if not (exactos or parciales):  # todas las palabras presentes, en cualquier orden
            palabras = [w for w in q.split("_") if len(w) > 1]
            parciales = [x for x in disp if palabras and all(w in D.normalizar(x) for w in palabras)]
        out.extend(exactos or parciales)
    return list(dict.fromkeys(out))


def _a_json(obj) -> str:
    def conv(o):
        if isinstance(o, pd.DataFrame):
            return o.head(60).to_dict("records")
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else round(float(o), 3)
        if isinstance(o, (pd.Timestamp, date)):
            return str(o)
        return str(o)

    def limpiar(o):
        if isinstance(o, dict):
            return {str(k): limpiar(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [limpiar(v) for v in o]
        if isinstance(o, pd.DataFrame):
            return limpiar(conv(o))
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, float):
            return None if np.isnan(o) else round(o, 3)
        return o
    return json.dumps(limpiar(obj), ensure_ascii=False, default=conv)


class SinDatos(Exception):
    pass


class Herramientas:
    def __init__(self, d: pd.DataFrame, capacidad: float):
        self.d = d
        self.capacidad = capacidad
        self.graficos: list = []   # figuras generadas para mostrar en el chat
        self.pasos: list = []      # registro de herramientas usadas

    # -- filtros ---------------------------------------------------------------
    def _filtrar(self, f: dict | None) -> pd.DataFrame:
        f = f or {}
        d = self.d
        kw = {}
        for clave, std in (("ubicaciones", "_ubic"), ("tipos", "_tipo"), ("sentidos", "_sent")):
            if f.get(clave) and D.tiene(d, std):
                kw[clave] = _emparejar(f[clave], d[std].dropna().unique()) or ["__ninguno__"]
        r = D.filtrar(d, desde=f.get("fecha_desde"), hasta=f.get("fecha_hasta"),
                      hora_desde=f.get("hora_desde"), hora_hasta=f.get("hora_hasta"), **kw)
        if f.get("dias_semana") and D.tiene(r, "_ts"):
            dias = [D.DIAS.index(x) for x in _emparejar(f["dias_semana"], D.DIAS)]
            r = r[r["_ts"].dt.dayofweek.isin(dias)]
        if r.empty:
            raise SinDatos(f"No hay datos con los filtros {f}.")
        return r

    # -- herramientas --------------------------------------------------------------
    def valores_disponibles(self, incluir_columnas: bool = True) -> dict:
        d = self.d
        r = {}
        for nombre, std in _STD.items():
            if D.tiene(d, std):
                r[nombre + "es" if nombre.endswith("n") else nombre + "s"] = sorted(map(str, d[std].dropna().unique()))[:80]
        if D.tiene(d, "_ts"):
            r["fecha_minima"], r["fecha_maxima"] = str(d["_ts"].min()), str(d["_ts"].max())
            r["tiene_hora"] = bool(D.tiene_hora(d))
        r["tiene_velocidad"] = bool(D.tiene(d, "_vel"))
        if incluir_columnas:
            r["columnas_originales"] = [c for c in d.columns if not c.startswith("_")]
        return r

    def resumen_general(self, filtros=None) -> dict:
        d = self._filtrar(filtros)
        return D.kpis(d) if len(d) else {"error": "No hay datos con esos filtros."}

    def consultar(self, agrupar_por=None, metrica="vehiculos", agregacion="suma",
                  orden="desc", limite=30, filtros=None):
        d = self._filtrar(filtros)
        if not len(d):
            return {"error": "No hay datos con esos filtros."}
        return D.consulta(d, agrupar_por, metrica, agregacion, orden, min(int(limite or 30), 200))

    def horas_pico(self, n=5, por_ubicacion=False, filtros=None):
        d = self._filtrar(filtros)
        return D.horas_pico(d, int(n or 5), "_ubic" if por_ubicacion and D.tiene(d, "_ubic") else None)

    def nivel_de_servicio(self, capacidad_veh_h=None, filtros=None):
        d = self._filtrar(filtros)
        cap = float(capacidad_veh_h or self.capacidad)
        if not D.tiene_hora(d):
            return {"error": "Se necesitan datos horarios para calcular v/c."}
        return {"capacidad_usada_veh_h": cap,
                "criterio": "A ≤0.60, B ≤0.70, C ≤0.80, D ≤0.90, E ≤1.00, F >1.00 (referencial)",
                "resultados": D.nivel_servicio(d, cap)}

    def detectar_anomalias(self, umbral_z=3.0, filtros=None):
        d = self._filtrar(filtros)
        a = D.anomalias(d, float(umbral_z or 3.0))
        return {"total_anomalias": len(a), "anomalias": a.head(25)}

    def pronosticar(self, horizonte=24, filtros=None):
        d = self._filtrar(filtros)
        r = D.pronostico(d, int(horizonte or 24))
        if "error" in r:
            return r
        self.graficos.append(G.pronostico(r))
        salida = {k: v for k, v in r.items() if k not in ("historia", "tabla")}
        t = r["tabla"].copy()
        t["fecha_hora"] = t["fecha_hora"].astype(str)
        salida["pronostico"] = t.round(0)
        salida["nota"] = "El gráfico del pronóstico ya se mostró al usuario."
        return salida

    def crear_grafico(self, grafico="perfil_horario", separar_por="ninguno", agrupar_por=None,
                      metrica="vehiculos", agregacion="suma", estilo="barras", titulo="", filtros=None):
        d = self._filtrar(filtros)
        if not len(d):
            return {"error": "No hay datos con esos filtros."}
        por = _STD.get(separar_por or "ninguno")
        por = por if por and D.tiene(d, por) else None
        horario = D.tiene_hora(d)
        try:
            if grafico == "serie_temporal":
                fig = G.serie_temporal(d, por)
            elif grafico == "perfil_horario" and horario:
                fig = G.perfil_horario(d, por)
            elif grafico == "perfil_semanal":
                fig = G.perfil_semanal(d)
            elif grafico == "mapa_calor" and horario:
                fig = G.mapa_calor(d)
            elif grafico == "por_tipo" and D.tiene(d, "_tipo"):
                fig = G.por_categoria(d, "_tipo", dona=estilo == "pastel")
            elif grafico == "por_ubicacion" and D.tiene(d, "_ubic"):
                fig = G.por_categoria(d, "_ubic", dona=estilo == "pastel")
            elif grafico == "composicion_por_hora" and horario and D.tiene(d, "_tipo"):
                fig = G.apilado_tipo_hora(d)
            elif grafico == "velocidad_vs_flujo" and D.tiene(d, "_vel"):
                fig = G.velocidad_vs_flujo(d)
            elif grafico == "nivel_servicio" and horario:
                fig = G.nivel_servicio(d, self.capacidad)
            elif grafico == "pronostico":
                r = D.pronostico(d)
                if "error" in r:
                    return r
                fig = G.pronostico(r)
            elif grafico == "consulta":
                t = D.consulta(d, agrupar_por or ["hora"], metrica, agregacion or "suma",
                               "natural" if estilo == "lineas" else "desc", 50)
                fig = G.desde_tabla(t, estilo or "barras", titulo)
            else:
                return {"error": f"No se puede generar '{grafico}' con las columnas disponibles."}
        except Exception as e:  # noqa: BLE001
            return {"error": f"No se pudo graficar: {e}"}
        if titulo:
            fig.update_layout(title=titulo)
        self.graficos.append(fig)
        return {"ok": True, "mensaje": "Gráfico mostrado al usuario."}

    # -- despachador -------------------------------------------------------------
    def ejecutar(self, nombre: str, args: dict | None) -> str:
        args = args or {}
        self.pasos.append({"herramienta": nombre, "argumentos": args})
        fn = getattr(self, nombre, None)
        if fn is None or nombre.startswith("_"):
            return _a_json({"error": f"Herramienta desconocida: {nombre}"})
        try:
            return _a_json(fn(**args))
        except SinDatos as e:
            return _a_json({"error": str(e), "valores_validos": self.valores_disponibles(False)})
        except TypeError as e:
            return _a_json({"error": f"Argumentos inválidos para {nombre}: {e}"})
        except Exception as e:  # noqa: BLE001
            return _a_json({"error": f"Fallo en {nombre}: {e}"})


# ----------------------------------------------------------------------------
# Instrucciones del agente
# ----------------------------------------------------------------------------
def instrucciones(h: Herramientas, tabla: str, filtros_activos: str) -> str:
    v = h.valores_disponibles(incluir_columnas=True)
    return f"""Eres "MoviAgente", un agente de IA especializado en ingeniería de tránsito y análisis de flujo vehicular.
Trabajas sobre la tabla '{tabla}' de Supabase. Hoy es {date.today():%d/%m/%Y}.

Contexto de los datos (ya cargados y con los filtros del panel aplicados: {filtros_activos}):
{json.dumps(v, ensure_ascii=False, default=str)[:3000]}

Reglas:
- Responde SIEMPRE en español, claro y directo, como un analista que le explica a un ingeniero industrial.
- Toda cifra que menciones debe salir de una herramienta. Nunca inventes datos. Si algo no se puede calcular con las columnas disponibles, dilo.
- Puedes llamar varias herramientas seguidas. Si el usuario pide ver o comparar algo, usa crear_grafico.
- Usa los nombres exactos de ubicaciones/tipos que aparecen arriba.
- Estructura sugerida: respuesta corta primero, luego 2-4 datos clave en viñetas y, si corresponde, recomendaciones prácticas
  (ajuste de ciclos semafóricos u ondas verdes, restricción de carga pesada en hora punta, control de paraderos informales,
  desvíos, refuerzo de agentes de tránsito, escalonamiento de horarios de entrada, etc.).
- El nivel de servicio por v/c es referencial (simplificado del HCM); acláralo cuando lo uses.
- Sé breve: no repitas tablas enteras, resume lo importante."""


# ----------------------------------------------------------------------------
# Bucle del agente con cada proveedor
# ----------------------------------------------------------------------------
def _bucle_anthropic(cfg, sistema, historial, pregunta, h, max_pasos=8) -> str:
    import anthropic
    cliente = anthropic.Anthropic(api_key=cfg["api_key"])
    msgs = historial + [{"role": "user", "content": pregunta}]
    texto = ""
    for _ in range(max_pasos):
        r = cliente.messages.create(model=cfg["modelo"], max_tokens=2500, system=sistema,
                                    tools=HERRAMIENTAS, messages=msgs)
        msgs.append({"role": "assistant", "content": r.content})
        texto = "".join(b.text for b in r.content if b.type == "text")
        if r.stop_reason != "tool_use":
            return texto
        resultados = [{"type": "tool_result", "tool_use_id": b.id, "content": h.ejecutar(b.name, b.input)}
                      for b in r.content if b.type == "tool_use"]
        msgs.append({"role": "user", "content": resultados})
    return texto or "Alcancé el límite de pasos. Intenta una pregunta más concreta."


def _bucle_openai(cfg, sistema, historial, pregunta, h, max_pasos=8) -> str:
    from openai import OpenAI
    cliente = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}} for t in HERRAMIENTAS]
    msgs = [{"role": "system", "content": sistema}] + historial + [{"role": "user", "content": pregunta}]
    for _ in range(max_pasos):
        r = cliente.chat.completions.create(model=cfg["modelo"], messages=msgs, tools=tools)
        m = r.choices[0].message
        if not m.tool_calls:
            return m.content or ""
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [tc.model_dump(exclude_none=True) for tc in m.tool_calls]})
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": h.ejecutar(tc.function.name, args)})
    return "Alcancé el límite de pasos. Intenta una pregunta más concreta."


def responder(pregunta: str, historial: list[dict], d: pd.DataFrame, cfg: dict,
              tabla: str = "", filtros_activos: str = "ninguno") -> dict:
    """
    cfg = {proveedor, modelo, api_key, capacidad}
    historial = [{"role": "user"|"assistant", "content": "texto"}, ...]
    Devuelve {"texto", "graficos", "pasos"}.
    """
    h = Herramientas(d, cfg.get("capacidad", 1800))
    if not cfg.get("api_key"):
        texto = agente_local(pregunta, h)
        return {"texto": texto, "graficos": h.graficos, "pasos": h.pasos}
    prov = PROVEEDORES[cfg["proveedor"]]
    conf = {**prov, **{k: v for k, v in cfg.items() if v}}
    sistema = instrucciones(h, tabla, filtros_activos)
    hist = [{"role": m["role"], "content": m["content"]} for m in historial[-10:] if m.get("content")]
    if prov["tipo"] == "anthropic":
        texto = _bucle_anthropic(conf, sistema, hist, pregunta, h)
    else:
        texto = _bucle_openai(conf, sistema, hist, pregunta, h)
    return {"texto": texto, "graficos": h.graficos, "pasos": h.pasos}


# ----------------------------------------------------------------------------
# Modo local (sin API key): reglas por palabras clave
# ----------------------------------------------------------------------------
def _n(x, dec=0):
    try:
        return f"{x:,.{dec}f}".replace(",", " ")
    except (TypeError, ValueError):
        return str(x)


def agente_local(pregunta: str, h: Herramientas) -> str:
    q = _sin_tildes(pregunta)
    d = h.d
    if not len(d):
        return "No hay datos con los filtros actuales."
    partes = []

    def hay(*palabras):
        return any(re.search(p, q) for p in palabras)

    if hay(r"pronost", r"predic", r"proyecc", r"futuro", r"manana", r"proxim"):
        r = h.pronosticar(24)
        if "error" in r:
            partes.append(r["error"])
        else:
            t = r["pronostico"]
            pico = t.loc[t["pronostico"].idxmax()]
            partes.append(f"**Pronóstico ({r['metodo']})**\n\n"
                          f"- Próximo pico esperado: **{pico['fecha_hora']}** con ~{_n(pico['pronostico'])} vehículos.\n"
                          f"- Tendencia diaria: {r['tendencia_diaria_pct']:+.2f} %.\n"
                          f"- Error de validación (MAPE): {r['mape_validacion_pct']} %.")
    if hay(r"anomal", r"inusual", r"raro", r"incident", r"atipic", r"accident", r"cierre"):
        r = h.detectar_anomalias(3.0)
        a = r["anomalias"]
        if len(a):
            filas = "\n".join(f"- {f['momento']}{' · ' + str(f['ubicacion']) if 'ubicacion' in f else ''}: "
                              f"{_n(f['vehiculos'])} veh. (esperado {_n(f['esperado'])}) → {f['tipo_anomalia']}"
                              for _, f in a.head(6).iterrows())
            partes.append(f"**Se detectaron {r['total_anomalias']} anomalías.** Las más fuertes:\n\n{filas}")
        else:
            partes.append("No se detectaron anomalías relevantes con el umbral actual.")
    if hay(r"servicio", r"capacidad", r"v/c", r"saturac", r"congest"):
        r = h.nivel_de_servicio()
        if "error" in r:
            partes.append(r["error"])
        else:
            t = r["resultados"]
            filas = "\n".join(f"- {f['ubicacion']}: v/c máx. {f['vc_maximo']} → nivel peor **{f['nivel_peor']}**, "
                              f"típico {f['nivel_tipico']} ({f['pct_horas_E_F']} % de horas en E/F)"
                              for _, f in t.iterrows())
            partes.append(f"**Nivel de servicio** (capacidad {_n(r['capacidad_usada_veh_h'])} veh/h, referencial):\n\n{filas}")
            h.crear_grafico("nivel_servicio")
    if hay(r"pico", r"punta", r"hora"):
        r = h.horas_pico(3)
        if "error" in r:
            partes.append(r["error"])
        else:
            hs = ", ".join(f"{f['hora']:02d}:00 (~{_n(f['promedio'])} veh/h)" for f in r["horas_del_dia_mas_cargadas"])
            partes.append(f"**Horas pico típicas:** {hs}.")
            h.crear_grafico("perfil_horario")
    if hay(r"tipo", r"composic", r"clase", r"vehiculo", r"auto", r"moto", r"bus", r"camion"):
        if D.tiene(d, "_tipo"):
            c = D.por_categoria(d, "_tipo", 6)
            partes.append("**Composición vehicular:** " + ", ".join(
                f"{f['categoria']} {f['porcentaje']:.1f} %" for _, f in c.iterrows()) + ".")
            h.crear_grafico("por_tipo", estilo="pastel")
    if hay(r"ubicac", r"intersec", r"calle", r"avenida", r"punto", r"zona", r"donde", r"cruce", r"via"):
        if D.tiene(d, "_ubic"):
            c = D.por_categoria(d, "_ubic", 5)
            partes.append("**Ubicaciones con más flujo:** " + ", ".join(
                f"{f['categoria']} ({_n(f['vehiculos'])} veh., {f['porcentaje']:.1f} %)" for _, f in c.iterrows()) + ".")
            h.crear_grafico("por_ubicacion")
    if hay(r"velocidad", r"rapid", r"lent"):
        if D.tiene(d, "_vel"):
            dim = "hora" if D.tiene_hora(d) else "dia_semana"
            v = D.consulta(d, [dim], "velocidad", "promedio", "asc", 3)
            etiqueta = (lambda x: f"{int(x):02d}:00") if dim == "hora" else str
            partes.append(f"**Velocidad media:** {_n(d['_vel'].mean(), 1)} km/h. Periodos más lentos: " +
                          ", ".join(f"{etiqueta(a)} ({_n(b, 1)} km/h)" for a, b in zip(v.iloc[:, 0], v.iloc[:, -1])) + ".")
            h.crear_grafico("velocidad_vs_flujo")
    if hay(r"semana", r"\bdia", r"lunes", r"viernes", r"sabado", r"domingo"):
        s = D.perfil_semanal(d)
        if len(s):
            f = s.loc[s["promedio"].idxmax()]
            g = s.loc[s["promedio"].idxmin()]
            partes.append(f"**Por día de la semana:** el más cargado es {f['dia']} (~{_n(f['promedio'])} veh/día) "
                          f"y el más ligero {g['dia']} (~{_n(g['promedio'])}).")
            h.crear_grafico("perfil_semanal")
    if not partes or hay(r"resumen", r"general", r"informe", r"kpi", r"situacion"):
        k = D.kpis(d)
        lineas = [f"- Total de vehículos: **{_n(k['total_vehiculos'])}** en {_n(k['registros'])} registros"]
        if "promedio_diario" in k:
            lineas.append(f"- Promedio diario: {_n(k['promedio_diario'])} vehículos ({k.get('dias_con_datos')} días)")
        for clave, rot in (("hora_pico", "Hora pico"), ("dia_semana_pico", "Día pico"),
                           ("ubicacion_mas_transitada", "Ubicación más transitada"),
                           ("tipo_predominante", "Tipo predominante")):
            if clave in k:
                lineas.append(f"- {rot}: {k[clave]}")
        if "velocidad_promedio" in k:
            lineas.append(f"- Velocidad media: {_n(k['velocidad_promedio'], 1)} km/h")
        partes.insert(0, "**Resumen del flujo vehicular**\n\n" + "\n".join(lineas))
    partes.append("_Modo local (sin IA): respuestas por palabras clave. Agrega una API key en la barra lateral "
                  "para conversar libremente con el agente._")
    return "\n\n".join(partes)


# ----------------------------------------------------------------------------
# Informe automático
# ----------------------------------------------------------------------------
def informe_markdown(d: pd.DataFrame, tabla: str, capacidad: float) -> str:
    k = D.kpis(d)
    md = [f"# Informe de flujo vehicular – {tabla}", f"Generado el {date.today():%d/%m/%Y}", "",
          "## 1. Indicadores generales"]
    etiquetas = {"registros": ("Registros", 0), "total_vehiculos": ("Total de vehículos", 0),
                 "fecha_inicio": ("Desde", None), "fecha_fin": ("Hasta", None),
                 "dias_con_datos": ("Días con datos", 0), "promedio_diario": ("Promedio diario (veh/día)", 0),
                 "dia_semana_pico": ("Día de la semana más cargado", None), "hora_pico": ("Hora pico", None),
                 "flujo_hora_pico_promedio": ("Flujo promedio en hora pico (veh/h)", 0),
                 "flujo_horario_maximo": ("Flujo horario máximo registrado (veh/h)", 0),
                 "momento_flujo_maximo": ("Momento del flujo máximo", None),
                 "ubicacion_mas_transitada": ("Ubicación más transitada", None),
                 "tipo_predominante": ("Tipo de vehículo predominante", None),
                 "velocidad_promedio": ("Velocidad media (km/h)", 1)}
    for clave, v in k.items():
        rot, dec = etiquetas.get(clave, (clave, None))
        md.append(f"- **{rot}:** {_n(v, dec) if dec is not None else v}")
    if D.tiene_hora(d):
        hp = D.horas_pico(d, 5)
        md += ["", "## 2. Horas pico", "| Hora | Promedio veh/h | Máximo |", "|---|---|---|"]
        md += [f"| {f['hora']:02d}:00 | {_n(f['promedio'])} | {_n(f['maximo'])} |"
               for f in hp["horas_del_dia_mas_cargadas"]]
        ns = D.nivel_servicio(d, capacidad)
        md += ["", f"## 3. Nivel de servicio (capacidad {_n(capacidad)} veh/h, referencial)",
               "| Ubicación | v/c máx. | Nivel típico | Nivel peor | % horas E/F |", "|---|---|---|---|---|"]
        md += [f"| {f['ubicacion']} | {f['vc_maximo']} | {f['nivel_tipico']} | {f['nivel_peor']} | {f['pct_horas_E_F']} |"
               for _, f in ns.iterrows()]
    for std, titulo in (("_tipo", "Composición vehicular"), ("_ubic", "Flujo por ubicación")):
        if D.tiene(d, std):
            c = D.por_categoria(d, std, 10)
            md += ["", f"## {titulo}", "| Categoría | Vehículos | % |", "|---|---|---|"]
            md += [f"| {f['categoria']} | {_n(f['vehiculos'])} | {f['porcentaje']:.1f} |" for _, f in c.iterrows()]
    a = D.anomalias(d, 3.0)
    md += ["", "## Anomalías detectadas", f"Se encontraron {len(a)} periodos atípicos (|z| ≥ 3)."]
    for _, f in a.head(10).iterrows():
        md.append(f"- {f['momento']}: {_n(f['vehiculos'])} veh. vs {_n(f['esperado'])} esperados ({f['tipo_anomalia']})")
    p = D.pronostico(d, 24)
    if "error" not in p:
        md += ["", "## Pronóstico", f"Método: {p['metodo']}. Error de validación (MAPE): {p['mape_validacion_pct']} %."]
    return "\n".join(md)
