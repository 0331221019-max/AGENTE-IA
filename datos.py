"""
datos.py
Carga de la tabla desde Supabase, detección automática de columnas
y funciones de análisis del flujo vehicular.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import requests

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
ZONA_HORARIA = "America/Lima"


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def normalizar(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


# ----------------------------------------------------------------------------
# Conexión a Supabase (API REST)
# ----------------------------------------------------------------------------
def base_rest(url: str) -> str:
    url = url.strip().rstrip("/")
    return url.split("/rest/v1")[0] + "/rest/v1"


def _headers(key: str) -> dict:
    return {"apikey": key.strip(), "Accept": "application/json"}


def listar_tablas(url: str, key: str) -> list[str]:
    """Intenta leer el esquema OpenAPI para listar las tablas expuestas."""
    try:
        r = requests.get(base_rest(url) + "/", headers=_headers(key), timeout=15)
        if r.ok:
            rutas = r.json().get("paths", {})
            return sorted(p.strip("/") for p in rutas
                          if p.strip("/") and not p.startswith("/rpc"))
    except Exception:
        pass
    return []


def _explicar_error(r: requests.Response, tabla: str) -> str:
    try:
        detalle = r.json()
        msg = detalle.get("message") or str(detalle)
        pista = detalle.get("hint") or ""
    except Exception:
        msg, pista = r.text[:300], ""
    if r.status_code in (401, 403):
        return f"Supabase rechazó la clave (HTTP {r.status_code}). Revisa la API key. Detalle: {msg}"
    if r.status_code == 404:
        texto = f"No existe la tabla '{tabla}' o no está expuesta en la API. {msg}"
        return texto + (f" Sugerencia de Supabase: {pista}" if pista else "")
    return f"Error HTTP {r.status_code}: {msg}"


def cargar_tabla(url: str, key: str, tabla: str, max_filas: int = 300_000,
                 lote: int = 1000) -> pd.DataFrame:
    """Descarga toda la tabla por páginas (PostgREST devuelve máx. ~1000 filas por pedido)."""
    endpoint = f"{base_rest(url)}/{requests.utils.quote(tabla.strip())}"
    filas: list[dict] = []
    offset = 0
    while offset < max_filas:
        r = requests.get(endpoint, headers=_headers(key),
                         params={"select": "*", "limit": lote, "offset": offset},
                         timeout=40)
        if r.status_code >= 400:
            raise RuntimeError(_explicar_error(r, tabla))
        datos = r.json()
        if not datos:
            break
        filas.extend(datos)
        offset += len(datos)
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# Conversión de fechas y horas
# ----------------------------------------------------------------------------
_RE_TZ = re.compile(r"\d{2}:\d{2}(:\d{2}(\.\d+)?)?\s*(Z|[+-]\d{2}(:?\d{2})?)$")


def parsear_fechas(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_convert(ZONA_HORARIA).dt.tz_localize(None)
        return out
    txt = s.astype("string").str.strip()
    muestra = txt.dropna().head(200)
    dia_primero = bool(muestra.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}").any())
    con_tz = bool(muestra.apply(lambda v: bool(_RE_TZ.search(v))).any()) if len(muestra) else False
    out = pd.to_datetime(txt, errors="coerce", dayfirst=dia_primero,
                         format="mixed", utc=con_tz)
    if con_tz:
        out = out.dt.tz_convert(ZONA_HORARIA).dt.tz_localize(None)
    return out


def parsear_hora(s: pd.Series) -> pd.Series:
    """Devuelve la hora del día como número decimal (7.5 = 07:30)."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9 and num.dropna().between(0, 24).all():
        return num.astype(float)
    txt = s.astype("string").str.lower()
    partes = txt.str.extract(r"(\d{1,2})\s*[:h.]\s*(\d{2})")
    h = pd.to_numeric(partes[0], errors="coerce")
    m = pd.to_numeric(partes[1], errors="coerce").fillna(0)
    solo_h = pd.to_numeric(txt.str.extract(r"^\s*(\d{1,2})\s*(?:am|pm|h)?\s*$")[0], errors="coerce")
    h = h.fillna(solo_h)
    pm = txt.str.contains(r"p\.?\s*m", na=False) & (h < 12)
    am12 = txt.str.contains(r"a\.?\s*m", na=False) & (h == 12)
    h = h.where(~pm, h + 12).where(~am12, 0)
    return (h + m / 60).where(h.between(0, 24))


# ----------------------------------------------------------------------------
# Detección automática de columnas
# ----------------------------------------------------------------------------
PALABRAS = {
    "fecha": ["fecha_hora", "fechahora", "timestamp", "datetime", "fecha_registro",
              "fecha", "date", "dia", "momento", "registrado", "created_at"],
    "hora": ["hora", "hour", "franja_horaria", "franja", "intervalo", "periodo", "time"],
    "cantidad": ["cantidad", "conteo", "n_vehiculos", "num_vehiculos", "numero_vehiculos",
                 "nro_vehiculos", "cantidad_vehiculos", "vehiculos", "volumen", "flujo",
                 "aforo", "intensidad", "total", "count"],
    "velocidad": ["velocidad_promedio", "velocidad", "speed", "km_h", "kmh"],
    "tipo": ["tipo_vehiculo", "tipo_de_vehiculo", "clase_vehiculo", "tipo", "clase",
             "categoria", "vehiculo", "vehicle"],
    "ubicacion": ["ubicacion", "interseccion", "punto", "punto_control", "calle", "avenida",
                  "via", "zona", "tramo", "cruce", "estacion", "sensor", "lugar",
                  "distrito", "location", "camara"],
    "sentido": ["sentido", "direccion", "carril", "direction", "lane"],
}

TIPOS_VEHICULO = ["auto", "automovil", "carro", "camioneta", "station", "pickup", "bus",
                  "omnibus", "micro", "combi", "coaster", "camion", "trailer", "moto",
                  "mototaxi", "taxi", "bicicleta", "liviano", "pesado", "particular",
                  "colectivo", "furgon", "volquete", "cisterna"]

ROLES = ["fecha", "hora", "cantidad", "velocidad", "tipo", "ubicacion", "sentido"]
NOMBRE_ROL = {"fecha": "Fecha / fecha-hora", "hora": "Hora (si va aparte)",
              "cantidad": "Cantidad de vehículos", "velocidad": "Velocidad",
              "tipo": "Tipo de vehículo", "ubicacion": "Ubicación / intersección",
              "sentido": "Sentido / carril"}


def _puntaje(col: str, palabras: list[str]) -> int:
    mejor = 0
    for i, p in enumerate(palabras):
        peso = len(palabras) - i
        if col == p:
            mejor = max(mejor, 100 + peso)
        elif col.startswith(p + "_") or col.endswith("_" + p):
            mejor = max(mejor, 60 + peso)
        elif p in col:
            mejor = max(mejor, 30 + peso)
    return mejor


def _es_id(col: str) -> bool:
    return col in ("id", "uuid", "pk") or col.endswith("_id") or col.startswith("id_")


def _es_numerica(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce")
    return len(s.dropna()) > 0 and v.notna().sum() >= 0.9 * len(s.dropna())


def _es_fecha(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return False
    muestra = s.dropna().head(300)
    if muestra.empty:
        return False
    v = parsear_fechas(muestra)
    return v.notna().mean() > 0.8 and v.dt.year.between(1990, 2100).mean() > 0.8


def _es_hora(s: pd.Series) -> bool:
    muestra = s.dropna().head(300)
    return not muestra.empty and parsear_hora(muestra).notna().mean() > 0.8


def _es_categoria(s: pd.Series, max_unicos: int = 300) -> bool:
    noval = s.dropna()
    if noval.empty:
        return False
    n = noval.astype(str).nunique()
    return n <= max_unicos and (n < 0.9 * len(noval) or len(noval) < 20)


def detectar_columnas(df: pd.DataFrame) -> dict:
    """Asigna cada rol (fecha, cantidad, tipo...) a la columna más probable."""
    cols = {c: normalizar(c) for c in df.columns}
    usados: set[str] = set()
    roles: dict = {r: None for r in ROLES}

    # ¿Formato "ancho"? (una columna numérica por tipo de vehículo: autos, buses, motos...)
    anchas = [c for c, n in cols.items()
              if not _es_id(n) and _es_numerica(df[c])
              and any(n == t or n.startswith(t) or n.rstrip("s") == t for t in TIPOS_VEHICULO)]
    roles["columnas_tipo"] = anchas if len(anchas) >= 2 else []
    usados.update(roles["columnas_tipo"])

    validadores = {
        "fecha": _es_fecha,
        "hora": _es_hora,
        "cantidad": lambda s: _es_numerica(s) and (pd.to_numeric(s, errors="coerce").dropna() >= 0).all(),
        "velocidad": _es_numerica,
        "tipo": _es_categoria,
        "ubicacion": _es_categoria,
        "sentido": lambda s: _es_categoria(s, 30),
    }
    for rol in ["fecha", "hora", "velocidad", "cantidad", "tipo", "ubicacion", "sentido"]:
        if rol == "tipo" and roles["columnas_tipo"]:
            continue
        candidatos = []
        for c, n in cols.items():
            if c in usados or (rol in ("cantidad", "velocidad") and _es_id(n)):
                continue
            p = _puntaje(n, PALABRAS[rol])
            if rol == "fecha" and n == "created_at":
                p = 5  # solo si no hay otra fecha
            if p > 0 and validadores[rol](df[c]):
                candidatos.append((p, c))
        if candidatos:
            roles[rol] = max(candidatos)[1]
            usados.add(roles[rol])

    if roles["columnas_tipo"]:
        roles["cantidad"] = None  # en formato ancho la cantidad sale de las columnas por tipo
    # Si la "fecha" ya trae la hora, no hace falta una columna de hora aparte
    if roles["fecha"] and roles["hora"]:
        f = parsear_fechas(df[roles["fecha"]].dropna().head(300))
        if ((f.dt.hour != 0) | (f.dt.minute != 0)).any():
            roles["hora"] = None
    return roles


def preparar(df: pd.DataFrame, roles: dict) -> pd.DataFrame:
    """Crea columnas estándar: _ts, _cant, _tipo, _ubic, _sent, _vel."""
    d = df.copy()
    anchas = [c for c in roles.get("columnas_tipo") or [] if c in d.columns]
    if anchas:
        otras = [c for c in d.columns if c not in anchas
                 and normalizar(c) not in ("total", "total_vehiculos")]
        d = d.melt(id_vars=otras, value_vars=anchas, var_name="_tipo", value_name="_cant_ancho")
        d["_cant"] = pd.to_numeric(d["_cant_ancho"], errors="coerce").fillna(0)
        d = d.drop(columns="_cant_ancho")
        d["_tipo"] = d["_tipo"].astype(str).str.replace("_", " ").str.strip().str.capitalize()
    else:
        c = roles.get("cantidad")
        d["_cant"] = pd.to_numeric(d[c], errors="coerce").fillna(0) if c else 1.0
        c = roles.get("tipo")
        d["_tipo"] = d[c].astype("string").str.strip() if c else pd.NA

    c = roles.get("fecha")
    d["_ts"] = parsear_fechas(d[c]) if c else pd.NaT
    c = roles.get("hora")
    if c:
        horas = parsear_hora(d[c])
        base = d["_ts"].dt.normalize() if roles.get("fecha") else pd.Timestamp("2000-01-03")
        d["_ts"] = base + pd.to_timedelta(horas, unit="h")
    for rol, std in (("ubicacion", "_ubic"), ("sentido", "_sent")):
        c = roles.get(rol)
        d[std] = d[c].astype("string").str.strip() if c else pd.NA
    c = roles.get("velocidad")
    d["_vel"] = pd.to_numeric(d[c], errors="coerce") if c else np.nan
    for std in ("_tipo", "_ubic", "_sent"):
        d[std] = d[std].astype("string").fillna("Sin dato") if d[std].notna().any() else pd.NA
    return d


def tiene(d: pd.DataFrame, std: str) -> bool:
    return std in d.columns and d[std].notna().any()


def solo_fecha_sin_hora(d: pd.DataFrame) -> bool:
    ts = d["_ts"].dropna()
    return ts.empty or not ((ts.dt.hour != 0) | (ts.dt.minute != 0)).any()


def tiene_hora(d: pd.DataFrame) -> bool:
    return tiene(d, "_ts") and not solo_fecha_sin_hora(d)


def filtrar(d: pd.DataFrame, desde=None, hasta=None, ubicaciones=None, tipos=None,
            sentidos=None, hora_desde=None, hora_hasta=None) -> pd.DataFrame:
    m = pd.Series(True, index=d.index)
    if desde is not None and tiene(d, "_ts"):
        m &= d["_ts"] >= pd.Timestamp(desde)
    if hasta is not None and tiene(d, "_ts"):
        m &= d["_ts"] < pd.Timestamp(hasta) + pd.Timedelta(days=1)
    if ubicaciones:
        m &= d["_ubic"].isin(ubicaciones)
    if tipos:
        m &= d["_tipo"].isin(tipos)
    if sentidos:
        m &= d["_sent"].isin(sentidos)
    if hora_desde is not None and tiene_hora(d):
        m &= d["_ts"].dt.hour >= int(hora_desde)
    if hora_hasta is not None and tiene_hora(d):
        m &= d["_ts"].dt.hour <= int(hora_hasta)
    return d[m.fillna(False)]


# ----------------------------------------------------------------------------
# Análisis
# ----------------------------------------------------------------------------
def flujo_horario(d: pd.DataFrame, por: str | None = None) -> pd.DataFrame:
    """Vehículos por cada hora registrada (solo horas con datos)."""
    x = d.dropna(subset=["_ts"]).copy()
    x["_h"] = x["_ts"].dt.floor("h")
    claves = ["_h"] + ([por] if por else [])
    return x.groupby(claves, observed=True)["_cant"].sum().reset_index()


def flujo_diario(d: pd.DataFrame, por: str | None = None) -> pd.DataFrame:
    x = d.dropna(subset=["_ts"]).copy()
    x["_d"] = x["_ts"].dt.normalize()
    claves = ["_d"] + ([por] if por else [])
    return x.groupby(claves, observed=True)["_cant"].sum().reset_index()


def perfil_horario(d: pd.DataFrame, por: str | None = None) -> pd.DataFrame:
    fh = flujo_horario(d, por)
    fh["hora"] = fh["_h"].dt.hour
    g = ["hora"] + ([por] if por else [])
    return (fh.groupby(g)["_cant"]
            .agg(promedio="mean", maximo="max", horas_observadas="count")
            .reset_index())


def perfil_semanal(d: pd.DataFrame) -> pd.DataFrame:
    fd = flujo_diario(d)
    fd["n"] = fd["_d"].dt.dayofweek
    r = fd.groupby("n")["_cant"].agg(promedio="mean", dias_observados="count").reset_index()
    r["dia"] = r["n"].map(dict(enumerate(DIAS)))
    return r[["n", "dia", "promedio", "dias_observados"]]


def matriz_dia_hora(d: pd.DataFrame) -> pd.DataFrame:
    fh = flujo_horario(d)
    fh["dia"] = fh["_h"].dt.dayofweek
    fh["hora"] = fh["_h"].dt.hour
    p = fh.pivot_table(index="dia", columns="hora", values="_cant", aggfunc="mean")
    p.index = [DIAS[i] for i in p.index]
    return p


def por_categoria(d: pd.DataFrame, std: str, top: int | None = None) -> pd.DataFrame:
    r = (d.groupby(std, observed=True)["_cant"].sum()
         .sort_values(ascending=False).reset_index())
    r.columns = ["categoria", "vehiculos"]
    total = r["vehiculos"].sum()
    r["porcentaje"] = (100 * r["vehiculos"] / total).round(2) if total else 0
    return r.head(top) if top else r


def kpis(d: pd.DataFrame) -> dict:
    k = {"registros": int(len(d)), "total_vehiculos": float(d["_cant"].sum())}
    if tiene(d, "_ts"):
        fd = flujo_diario(d)
        k["fecha_inicio"] = str(d["_ts"].min())
        k["fecha_fin"] = str(d["_ts"].max())
        k["dias_con_datos"] = int(fd["_d"].nunique())
        k["promedio_diario"] = float(fd.groupby("_d")["_cant"].sum().mean()) if len(fd) else 0.0
        sem = perfil_semanal(d)
        if len(sem):
            fila = sem.loc[sem["promedio"].idxmax()]
            k["dia_semana_pico"] = fila["dia"]
        if tiene_hora(d):
            ph = perfil_horario(d)
            fila = ph.loc[ph["promedio"].idxmax()]
            k["hora_pico"] = f"{int(fila['hora']):02d}:00 - {int(fila['hora']) + 1:02d}:00"
            k["flujo_hora_pico_promedio"] = float(fila["promedio"])
            fh = flujo_horario(d)
            k["flujo_horario_maximo"] = float(fh["_cant"].max())
            k["momento_flujo_maximo"] = str(fh.loc[fh["_cant"].idxmax(), "_h"])
    if tiene(d, "_ubic"):
        k["ubicacion_mas_transitada"] = por_categoria(d, "_ubic").iloc[0]["categoria"]
    if tiene(d, "_tipo"):
        t = por_categoria(d, "_tipo").iloc[0]
        k["tipo_predominante"] = f"{t['categoria']} ({t['porcentaje']:.1f}%)"
    if tiene(d, "_vel"):
        k["velocidad_promedio"] = float(d["_vel"].mean())
    return k


def horas_pico(d: pd.DataFrame, n: int = 5, por: str | None = None) -> dict:
    if not tiene_hora(d):
        return {"error": "La tabla no tiene información horaria."}
    ph = perfil_horario(d, por).sort_values("promedio", ascending=False)
    fh = flujo_horario(d, por).sort_values("_cant", ascending=False).head(n)
    fh["_h"] = fh["_h"].astype(str)
    return {"horas_del_dia_mas_cargadas": ph.head(n if not por else n * 3).round(1).to_dict("records"),
            "momentos_puntuales_con_mas_flujo": fh.rename(columns={"_h": "fecha_hora", "_cant": "vehiculos"}).to_dict("records")}


NIVELES = [(0.60, "A", "Flujo libre"), (0.70, "B", "Flujo estable"),
           (0.80, "C", "Flujo estable con restricciones"), (0.90, "D", "Próximo a inestable"),
           (1.00, "E", "Capacidad (inestable)"), (np.inf, "F", "Congestión / colapso")]


def nivel_de(vc: float) -> str:
    for lim, letra, _ in NIVELES:
        if vc <= lim:
            return letra
    return "F"


def nivel_servicio(d: pd.DataFrame, capacidad: float, por: str | None = "_ubic") -> pd.DataFrame:
    """Relación volumen/capacidad (v/c) por hora y nivel de servicio referencial A–F."""
    por = por if por and tiene(d, por) else None
    fh = flujo_horario(d, por)
    fh["vc"] = fh["_cant"] / capacidad
    fh["nivel"] = fh["vc"].apply(nivel_de)
    g = fh.groupby(por) if por else fh.assign(_todo="General").groupby("_todo")
    filas = []
    for clave, sub in g:
        i = sub["_cant"].idxmax()
        filas.append({
            "ubicacion": clave,
            "flujo_hora_promedio": round(sub["_cant"].mean(), 1),
            "flujo_hora_maximo": round(sub["_cant"].max(), 1),
            "vc_promedio": round(sub["vc"].mean(), 3),
            "vc_maximo": round(sub["vc"].max(), 3),
            "nivel_tipico": nivel_de(sub["vc"].median()),
            "nivel_peor": nivel_de(sub["vc"].max()),
            "pct_horas_E_F": round(100 * sub["nivel"].isin(["E", "F"]).mean(), 1),
            "hora_mas_critica": str(sub.loc[i, "_h"]),
        })
    return pd.DataFrame(filas).sort_values("vc_maximo", ascending=False)


def anomalias(d: pd.DataFrame, umbral_z: float = 2.5, por: str | None = "_ubic") -> pd.DataFrame:
    """Horas cuyo flujo se aleja mucho de lo normal para esa hora (y día, si hay datos)."""
    por = por if por and tiene(d, por) else None
    if not tiene_hora(d):
        x = flujo_diario(d, por)
        x["clave"] = x["_d"].dt.dayofweek.astype(str)
        tcol = "_d"
    else:
        x = flujo_horario(d, por)
        semanas = (x["_h"].max() - x["_h"].min()).days / 7
        x["clave"] = (x["_h"].dt.dayofweek.astype(str) + "-" + x["_h"].dt.hour.astype(str)
                      if semanas >= 3 else x["_h"].dt.hour.astype(str))
        tcol = "_h"
    grupo = ([por] if por else []) + ["clave"]
    x["_c2"] = x["_cant"] ** 2
    est = x.groupby(grupo, observed=True).agg(s1=("_cant", "sum"), s2=("_c2", "sum"), n=("_cant", "count"))
    x = x.join(est, on=grupo)
    x = x[x["n"] >= 4].copy()
    # Se compara cada periodo contra los DEMÁS periodos equivalentes (deja-uno-fuera),
    # así un valor atípico no "infla" su propia referencia.
    m = x["n"] - 1
    x["mean"] = (x["s1"] - x["_cant"]) / m
    var = (x["s2"] - x["_c2"] - m * x["mean"] ** 2) / (m - 1)
    sd = np.sqrt(var.clip(lower=0))
    # Desviación mínima: 5 % del valor esperado (evita alertas por ruido mínimo)
    x["z"] = (x["_cant"] - x["mean"]) / np.maximum(sd, 0.05 * x["mean"]).replace(0, np.nan)
    r = x[x["z"].abs() >= umbral_z].copy()
    r["tipo_anomalia"] = np.where(r["z"] > 0, "Flujo inusualmente alto", "Flujo inusualmente bajo")
    r = r.rename(columns={tcol: "momento", "_cant": "vehiculos", "mean": "esperado", por or "_": "ubicacion"})
    r["momento"] = r["momento"].astype(str)
    cols = ["momento"] + (["ubicacion"] if por else []) + ["vehiculos", "esperado", "z", "tipo_anomalia"]
    return r[cols].round(2).sort_values("z", key=abs, ascending=False)


def pronostico(d: pd.DataFrame, horizonte: int = 24) -> dict:
    """
    Pronóstico por perfil típico (promedio por día de la semana y hora) ajustado
    por la tendencia de los totales diarios. Incluye validación con el último 20 %.
    """
    if not tiene(d, "_ts"):
        return {"error": "Se necesita una columna de fecha para pronosticar."}
    horario = tiene_hora(d)
    serie = flujo_horario(d) if horario else flujo_diario(d)
    t = "_h" if horario else "_d"
    serie = serie.sort_values(t)
    if len(serie) < 12:
        return {"error": "Hay muy pocos datos para pronosticar (mínimo 12 periodos)."}
    dias = (serie[t].max() - serie[t].min()).days + 1
    usar_dia = dias >= 14 or not horario

    def clave(ts):
        if horario:
            return (ts.dt.dayofweek.astype(str) + "-" + ts.dt.hour.astype(str)) if usar_dia else ts.dt.hour.astype(str)
        return ts.dt.dayofweek.astype(str)

    def ajustar(hist):
        hist = hist.assign(k=clave(hist[t]))
        perfil = hist.groupby("k")["_cant"].agg(["mean", "std"])
        perfil["std"] = perfil["std"].fillna(perfil["mean"] * 0.15)
        diario = hist.groupby(hist[t].dt.normalize())["_cant"].sum()
        pendiente = 0.0
        if len(diario) >= 7:
            xx = (diario.index - diario.index[0]).days.values.astype(float)
            pendiente = np.polyfit(xx, diario.values, 1)[0] / max(diario.mean(), 1e-9)
        return perfil, pendiente, hist[t].max()

    def predecir(perfil, pendiente, ultimo, fechas):
        k = clave(pd.Series(fechas))
        base = perfil.reindex(k.values)
        dias_adelante = (pd.Series(fechas) - ultimo).dt.total_seconds().values / 86400
        factor = np.clip(1 + pendiente * dias_adelante, 0.7, 1.3)
        return base["mean"].values * factor, base["std"].values * factor

    # Validación: entrenar con el 80 % y comparar contra el 20 % final
    corte = int(len(serie) * 0.8)
    entren, prueba = serie.iloc[:corte], serie.iloc[corte:]
    perfil, pend, ult = ajustar(entren)
    yhat, _ = predecir(perfil, pend, ult, prueba[t].values)
    real = prueba["_cant"].values
    ok = ~np.isnan(yhat) & (real > 0)
    mape = float(np.mean(np.abs((real[ok] - yhat[ok]) / real[ok])) * 100) if ok.any() else None
    mae = float(np.mean(np.abs(real[ok] - yhat[ok]))) if ok.any() else None

    # Pronóstico real con toda la historia
    perfil, pend, ult = ajustar(serie)
    paso = pd.Timedelta(hours=1) if horario else pd.Timedelta(days=1)
    futuras = pd.date_range(ult + paso, periods=horizonte, freq=paso)
    yhat, sd = predecir(perfil, pend, ult, futuras)
    pron = pd.DataFrame({"fecha_hora": futuras, "pronostico": yhat,
                         "minimo": np.maximum(yhat - 1.28 * sd, 0), "maximo": yhat + 1.28 * sd})
    pron = pron.dropna(subset=["pronostico"])
    return {
        "granularidad": "hora" if horario else "día",
        "metodo": ("perfil promedio por día de la semana y hora" if usar_dia and horario
                   else "perfil promedio por hora del día" if horario
                   else "perfil promedio por día de la semana") + " ajustado por tendencia",
        "tendencia_diaria_pct": round(pend * 100, 2),
        "mape_validacion_pct": None if mape is None else round(mape, 1),
        "mae_validacion": None if mae is None else round(mae, 1),
        "historia": serie.rename(columns={t: "fecha_hora", "_cant": "vehiculos"}),
        "tabla": pron,
    }


# ----------------------------------------------------------------------------
# Consulta genérica (la usa el agente)
# ----------------------------------------------------------------------------
DIMENSIONES = {"hora": "hora del día (0-23)", "dia_semana": "Lunes..Domingo",
               "fecha": "día calendario", "semana": "semana del año", "mes": "mes",
               "tipo": "tipo de vehículo", "ubicacion": "ubicación", "sentido": "sentido/carril"}


def _dimension(d: pd.DataFrame, nombre: str) -> pd.Series:
    ts = d["_ts"]
    if nombre == "hora":
        return ts.dt.hour
    if nombre == "dia_semana":
        return pd.Categorical(ts.dt.dayofweek.map(dict(enumerate(DIAS))), categories=DIAS, ordered=True)
    if nombre == "fecha":
        return ts.dt.date.astype(str)
    if nombre == "semana":
        return ts.dt.strftime("%G-S%V")
    if nombre == "mes":
        return ts.dt.strftime("%Y-%m")
    std = {"tipo": "_tipo", "ubicacion": "_ubic", "sentido": "_sent"}.get(nombre)
    if std:
        return d[std]
    if nombre in d.columns:
        return d[nombre]
    raise ValueError(f"No conozco la dimensión '{nombre}'. Opciones: {list(DIMENSIONES)} o una columna original.")


def consulta(d: pd.DataFrame, agrupar_por: list[str] | None = None, metrica: str = "vehiculos",
             agregacion: str = "suma", orden: str = "desc", limite: int = 30) -> pd.DataFrame:
    agrupar_por = agrupar_por or []
    if metrica == "velocidad":
        valor, agregacion = d["_vel"], "promedio" if agregacion == "suma" else agregacion
    elif metrica == "registros":
        valor, agregacion = pd.Series(1, index=d.index), "suma"
    else:
        valor = d["_cant"]
    fn = {"suma": "sum", "promedio": "mean", "maximo": "max", "minimo": "min", "conteo": "count"}[agregacion]
    if not agrupar_por:
        return pd.DataFrame({f"{metrica}_{agregacion}": [getattr(valor, fn)()]})
    claves = {g: _dimension(d, g) for g in agrupar_por}
    tabla = pd.DataFrame(claves).assign(_v=valor.values)
    r = tabla.groupby(list(claves), observed=True)["_v"].agg(fn).reset_index()
    r = r.rename(columns={"_v": f"{metrica}_{agregacion}"})
    col = r.columns[-1]
    if orden == "natural":
        r = r.sort_values(list(claves))
    else:
        r = r.sort_values(col, ascending=orden == "asc")
    return r.head(limite).round(2)


# ----------------------------------------------------------------------------
# Datos de ejemplo (para probar sin conexión)
# ----------------------------------------------------------------------------
def datos_ejemplo(dias: int = 28, semilla: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    ubic = {"Av. Principal / Jr. Comercio": 1.3, "Av. Industrial / Ca. Los Pinos": 0.7,
            "Panamericana Norte (acceso)": 1.9}
    tipos = {"Auto": 0.45, "Mototaxi": 0.2, "Combi": 0.12, "Camión": 0.08, "Bus": 0.05, "Moto": 0.1}
    curva = np.array([.15, .1, .08, .08, .15, .4, .8, 1.0, .9, .7, .65, .7, .8, .75, .65, .7,
                      .8, .95, 1.0, .85, .6, .45, .3, .2])
    filas = []
    inicio = pd.Timestamp.now().normalize() - pd.Timedelta(days=dias)
    for dd in range(dias):
        dia = inicio + pd.Timedelta(days=dd)
        f_dia = 0.75 if dia.dayofweek == 6 else 0.9 if dia.dayofweek == 5 else 1.0
        for h in range(5, 23):
            for u, fu in ubic.items():
                for sentido in ("Norte-Sur", "Sur-Norte"):
                    for tp, ft in tipos.items():
                        lam = 420 * curva[h] * f_dia * fu * ft * (1 + 0.003 * dd)
                        # Incidentes simulados: cierre parcial de vía y un evento masivo
                        if dd == dias - 9 and 14 <= h <= 17 and u.startswith("Av. Industrial"):
                            lam *= 0.25
                        if dd == dias - 5 and 19 <= h <= 21 and u.startswith("Av. Principal"):
                            lam *= 1.9
                        cant = int(rng.poisson(lam))
                        vel = max(8, 60 - 38 * curva[h] * f_dia * fu + rng.normal(0, 4))
                        filas.append({"id": len(filas) + 1,
                                      "fecha_hora": (dia + pd.Timedelta(hours=h)).isoformat(),
                                      "ubicacion": u, "sentido": sentido, "tipo_vehiculo": tp,
                                      "cantidad": cant, "velocidad_promedio": round(vel, 1)})
    return pd.DataFrame(filas)
