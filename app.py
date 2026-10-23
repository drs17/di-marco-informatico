"""
Di Marco Informático — Analizador narrativo heurístico.
Basado en criterios públicos de Marcelo Di Marco.
Herramienta de diagnóstico, no de juicio. Vos decidís qué cortar.
"""
import re
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import spacy
import streamlit as st

# ============ CONFIG ============
st.set_page_config(page_title="Di Marco Informático", page_icon="✍️", layout="wide")

# ============ CARGA DEL MODELO ============
@st.cache_resource
def cargar_modelo():
    try:
        return spacy.load("es_core_news_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm"], check=True)
        return spacy.load("es_core_news_sm")

try:
    nlp = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar spaCy: {e}")
    st.stop()

# ============ CONSTANTES ============
FILTROS = {
    "sentir": ["sentir", "siento", "siente", "sentía", "sentí", "sintió", "sintiera", "sentido"],
    "ver": ["veo", "veía", "vio", "viera", "visto"],
    "pensar": ["pensar", "pienso", "piensa", "pensaba", "pensó", "pensara"],
    "notar": ["notar", "noto", "nota", "notaba", "notó"],
    "observar": ["observar", "observo", "observa", "observaba", "observó"],
    "escuchar": ["escuchar", "escucho", "escucha", "escuchaba", "escuchó"],
    "parecer": ["parecer", "parece", "parecía", "pareció"],
    "recordar": ["recordar", "recuerdo", "recuerda", "recordaba", "recordó"],
    "advertir": ["advertir", "advierto", "advierte", "advertía", "advirtió"],
    "comprender": ["comprender", "comprendo", "comprende", "comprendía", "comprendió"],
}

CLICHES = [
    "a la vuelta de la esquina", "de la noche a la mañana", "más vale tarde que nunca",
    "sin sombra de duda", "en el último momento", "de repente", "de pronto",
    "de golpe", "de una vez por todas", "a fin de cuentas", "al fin y al cabo",
    "llorar a mares", "corazón de piedra", "frío como el hielo",
    "la curiosidad mató al gato", "no hay mal que por bien no venga",
    "el mundo se le venía encima", "se le heló la sangre",
    "brillaba como el sol", "dormía como un tronco", "corría como el viento",
]

EXPLICATIVOS = [
    "es decir", "o sea", "en otras palabras", "dicho de otro modo",
    "como si", "lo que es", "quiero decir", "vale decir",
]

# ============ ANÁLISIS ============
def analizar(texto):
    doc = nlp(texto)
    ors = list(doc.sents)
    parrs = [p.strip() for p in texto.split("\n\n") if p.strip()]
    if len(parrs) < 3:
        parrs = [p.strip() for p in texto.split("\n") if p.strip()]
    return {"doc": doc, "oraciones": ors, "parrafos": parrs, "texto": texto}

def metricas_basicas(a):
    ors = a["oraciones"]
    long_or = [len([t for t in o if t.is_alpha]) for o in ors]
    lemas = [t.lemma_.lower() for t in a["doc"] if t.is_alpha]
    return {
        "n_oraciones": len(ors),
        "n_palabras": sum(long_or),
        "n_parrafos": len(a["parrafos"]),
        "long_or": long_or,
        "long_prom": statistics.mean(long_or) if long_or else 0,
        "long_med": statistics.median(long_or) if long_or else 0,
        "long_min": min(long_or) if long_or else 0,
        "long_max": max(long_or) if long_or else 0,
        "desvio": statistics.stdev(long_or) if len(long_or) > 1 else 0,
        "riqueza": len(set(lemas)) / len(lemas) if lemas else 0,
    }

def detectar_dialogo(texto):
    lineas = texto.split("\n")
    con_d = [l for l in lineas if re.match(r"^\s*[—–\-]", l.strip())]
    return len(con_d), len(lineas)

def detectar_filtros(a):
    hallazgos = []
    for t in a["doc"]:
        lemma = t.lemma_.lower()
        for cat, formas in FILTROS.items():
            if lemma in formas:
                start = max(0, t.i - 8)
                end = min(len(a["doc"]), t.i + 9)
                hallazgos.append({
                    "cat": cat, "palabra": t.text,
                    "contexto": a["doc"][start:end].text,
                })
                break
    return hallazgos

def detectar_adverbios_mente(a):
    return [t for t in a["doc"] if t.text.lower().endswith("mente") and t.pos_ == "ADV"]

def detectar_adjetivos(a):
    return [t for t in a["doc"] if t.pos_ == "ADJ"]

def detectar_ecos(a, ventana=60, min_ap=3):
    grupos = defaultdict(list)
    for t in a["doc"]:
        if t.is_alpha and len(t.text) > 3:
            grupos[t.lemma_.lower()].append(t.i)
    ecos = []
    for lema, posiciones in grupos.items():
        if len(posiciones) >= min_ap:
            for i in range(len(posiciones) - min_ap + 1):
                v = posiciones[i:i + min_ap]
                if v[-1] - v[0] <= ventana:
                    ecos.append({"lema": lema, "n": len(posiciones), "primera": v[0]})
                    break
    return sorted(ecos, key=lambda x: -x["n"])

def detectar_rimas(a, ventana=15):
    palabras = [(t.text.lower(), t.i) for t in a["doc"] if t.is_alpha and len(t.text) >= 4]
    rimas = []
    sufijos_ignorar = ["ando", "iendo", "ado", "ido", "aba", "ía", "aron", "ieron", "amos", "emos", "imos", "ción", "sión", "mente"]
    
    for i, (w1, p1) in enumerate(palabras):
        for w2, p2 in palabras[i + 1:i + 10]:
            if p2 - p1 > ventana:
                break
            if w1[-3:] == w2[-3:] and w1 != w2:
                if any(w1.endswith(suf) for suf in sufijos_ignorar):
                    continue
                rimas.append({"a": w1, "b": w2, "dist": p2 - p1})
    return rimas

def detectar_cliches(texto):
    t_low = texto.lower()
    return [c for c in CLICHES if c in t_low]

def detectar_pasivas(a):
    resultado = []
    tokens = a["doc"]
    for i, t in enumerate(tokens):
        if t.lemma_.lower() == "ser" and t.pos_ == "AUX":
            for j in range(i + 1, min(i + 4, len(tokens))):
                if tokens[j].pos_ == "VERB" and re.search(r"(ado|ido|to|so|cho)$", tokens[j].text.lower()):
                    resultado.append({"frase": tokens[i:j + 1].text, "pos": i})
                    break
    return resultado

def detectar_explicativos(texto):
    t_low = texto.lower()
    return [e for e in EXPLICATIVOS if e in t_low]

def mapa_tension(a, n=12):
    ors = a["oraciones"]
    if len(ors) < 3:
        return []
    n = min(n, len(ors))
    tam = max(1, len(ors) // n)
    segs = []
    for i in range(n):
        ini = i * tam
        fin = (i + 1) * tam if i < n - 1 else len(ors)
        seg = ors[ini:fin]
        if not seg:
            continue
        tokens = [t for o in seg for t in o]
        pals = [t for t in tokens if t.is_alpha]
        long_prom = statistics.mean([len([t for t in o if t.is_alpha]) for o in seg])
        tiene_d = any(re.match(r"^\s*[—–\-]", o.text.strip()) for o in seg)
        n_adj = sum(1 for t in tokens if t.pos_ == "ADJ")
        dens_adj = n_adj / max(len(pals), 1)
        n_filt = sum(1 for t in tokens if t.lemma_.lower() in [f for fs in FILTROS.values() for f in fs])
        tension = (30 / max(long_prom, 1)) + (3 if tiene_d else 0) - (dens_adj * 15) - (n_filt * 0.3)
        segs.append({
            "n": i + 1, "tension": tension, "long_prom": long_prom,
            "tiene_dialogo": tiene_d, "dens_adj": dens_adj, "n_filt": n_filt,
            "preview": seg[0].text[:120],
        })
    return segs

def veredicto(a, metricas, segs):
    checks = []
    personas = set()
    palabras_ignorar = ["dios", "basuras", "culpa", "vos", "sólo", "apagó", "sopló", "sonreíme", "metí", "nati", "chenille"]
    
    for e in a["doc"].ents:
        if e.label_ == "PER":
            texto_ent = e.text.strip()
            if texto_ent.lower() not in palabras_ignorar:
                if any(t.pos_ == "PROPN" and t.text[0].isupper() for t in e) and not any(t.pos_ == "VERB" for t in e):
                    personas.add(texto_ent)

    checks.append({
        "nombre": "Personajes identificables",
        "ok": len(personas) >= 2,
        "detalle": f"Detectados: {', '.join(sorted(personas)) if personas else 'ninguno'}",
        "di_marco": "Sin personajes claros no hay conflicto. Si no hay dos voluntades enfrentadas, no hay cuento.",
    })

    n_d, n_l = detectar_dialogo(a["texto"])
    ratio = n_d / max(n_l, 1)
    checks.append({
        "nombre": "Presencia de diálogo",
        "ok": ratio > 0.05,
        "detalle": f"{n_d} líneas de diálogo de {n_l} totales ({ratio * 100:.1f}%)",
        "di_marco": "El diálogo es escena pura. Un texto sin diálogo puede ser un resumen, no un cuento.",
    })

    cv = metricas["desvio"] / max(metricas["long_prom"], 1)
    checks.append({
        "nombre": "Variación de ritmo",
        "ok": cv > 0.3,
        "detalle": f"Coef. de variación: {cv:.2f} (deseable > 0.30)",
        "di_marco": "Si todas las oraciones miden lo mismo, el lector se duerme. El ritmo nace de la variación.",
    })

    if segs:
        con_d = sum(1 for s in segs if s["tiene_dialogo"])
        r = con_d / len(segs)
        checks.append({
            "nombre": "Proporción escena/resumen",
            "ok": r > 0.3,
            "detalle": f"{con_d} de {len(segs)} segmentos con diálogo ({r * 100:.0f}%)",
            "di_marco": "La escena muestra; el resumen cuenta. Si todo es resumen, el lector no está ahí.",
        })

    if len(segs) > 3:
        tens = [s["tension"] for s in segs]
        pico = max(tens)
        prom = statistics.mean(tens)
        idx = tens.index(pico) + 1
        checks.append({
            "nombre": "Curva de tensión con clímax",
            "ok": pico > prom * 1.3,
            "detalle": f"Pico en segmento {idx} de {len(segs)} (pico: {pico:.1f}, prom: {prom:.1f})",
            "di_marco": "El efecto necesita un punto de máxima tensión. Si la curva es plana, no hay cuento: hay anécdota.",
        })
    return checks


# ============ UI ============
st.title("✍️ Di Marco Informático")
st.caption("Analizador narrativo heurístico basado en criterios públicos de Marcelo Di Marco. No lo sustituye: lo aplica.")

with st.sidebar:
    st.header("📄 Tu texto")
    texto = st.text_area(
        "Pegá tu cuento o fragmento:",
        height=400,
        placeholder="Pegá acá tu texto...",
    )
    st.divider()
    st.caption("⚙️ **Cómo funciona**")
    st.caption("1. Métricas objetivas: ritmo, repeticiones, densidad.")
    st.caption("2. Heurísticas Di Marco: contexto, santa tijera, efecto.")
    st.caption("3. Vos decidís qué cortar. La app no juzga.")
    st.divider()
    if st.button("🔄 Limpiar caché"):
        st.cache_resource.clear()
        st.rerun()

if not texto or len(texto.split()) < 30:
    st.info("👈 Pegá un texto de al menos 30 palabras para empezar.")
    with st.expander("¿Qué analiza esta herramienta?"):
        st.markdown("""
        **🎯 ¿Hay cuento?** — Checklist estructural: personajes, diálogo, ritmo, escena, clímax.

        **📊 Ritmo y tensión** — Gráfico de longitud de oraciones y mapa de tensión por segmento.

        **🗡️ Santa Tijera** — Filtros, voz pasiva, clichés, explicaciones de más.

        **🔁 Ecos y rimas** — Palabras que resuenan a corta distancia, rimas no buscadas.

        **🎭 Efectismo** — Adjetivos, adverbios en -mente, densidad descriptiva.

        **📝 Reporte** — Todo junto, descargable en Markdown.
        
        **🖼️ Imaginismo** — Detecta abstracciones y verbos sensoriales.
        """)
    st.stop()

if len(texto.split()) > 8000:
    st.warning("⚠️ Texto largo: el análisis puede demorar. Si tarda demasiado, probá con un fragmento.")

with st.spinner("Aplicando criterios Di Marco..."):
    a = analizar(texto)
    metricas = metricas_basicas(a)
    segs = mapa_tension(a)
    checks = veredicto(a, metricas, segs)

cols = st.columns(6)
cols[0].metric("Palabras", f"{metricas['n_palabras']:,}")
cols[1].metric("Oraciones", metricas["n_oraciones"])
cols[2].metric("Párrafos", metricas["n_parrafos"])
cols[3].metric("Long. oración", f"{metricas['long_prom']:.1f}")
cols[4].metric("Riqueza", f"{metricas['riqueza'] * 100:.0f}%")
cols[5].metric("Desvío", f"{metricas['desvio']:.1f}")

st.divider()

# --- AQUÍ ESTÁ EL ARREGLO: AGREGAMOS LA 7MA PESTAÑA A LA LISTA ---
tabs = st.tabs(["🎯 ¿Hay cuento?", "📊 Ritmo y tensión", "🗡️ Santa Tijera",
                "🔁 Ecos y rimas", "🎭 Efectismo", "📝 Reporte", "🖼️ Imaginismo"])

# ---- Tab 0 ----
with tabs[0]:
    st.header("🎯 ¿Hay cuento?")
    st.caption("Checklist heurística. No es veredicto: es mapa.")
    pasados = sum(1 for c in checks if c["ok"])
    total = len(checks)
    score = pasados / total if total else 0
    if score >= 0.8:
        st.success(f"**{pasados}/{total}** — Estructura sólida. Hay cuento.")
    elif score >= 0.5:
        st.warning(f"**{pasados}/{total}** — Hay material, pero hay agujeros. Trabajá los checks en rojo.")
    else:
        st.error(f"**{pasados}/{total}** — Todavía no es un cuento. Es un fragmento, una anécdota o un resumen.")
    st.progress(score)
    st.divider()
    for c in checks:
        icon = "✅" if c["ok"] else "❌"
        with st.expander(f"{icon} **{c['nombre']}** — {c['detalle']}"):
            st.info(f"**Criterio Di Marco:** {c['di_marco']}")

# ---- Tab 1 ----
with tabs[1]:
    st.header("📊 Ritmo y tensión")
    st.subheader("Longitud de cada oración")
    if metricas["long_or"]:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(range(1, len(metricas["long_or"]) + 1), metricas["long_or"], color="#2980b9", width=1)
        ax.axhline(metricas["long_prom"], color="red", linestyle="--", linewidth=1.5,
                   label=f"Promedio: {metricas['long_prom']:.1f}")
        ax.set_xlabel("Oración #")
        ax.set_ylabel("Palabras")
        ax.set_title("Si esta línea parece el horizonte del mar, tu ritmo está muerto.")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        cv = metricas["desvio"] / max(metricas["long_prom"], 1)
        if cv < 0.3:
            st.error("🔴 **Ritmo monótono.** Di Marco: *'Si todas las oraciones miden lo mismo, el lector se duerme.'*")
        elif metricas["long_max"] > metricas["long_prom"] * 2.5:
            st.warning("🟡 **Oraciones demasiado largas.** Hay picos de más del doble del promedio. Cortalas en dos.")
        else:
            st.success("✅ Ritmo variado.")

    st.divider()
    st.subheader("Mapa de tensión")
    st.caption("Tensión = oraciones cortas + diálogo + acción − adjetivos − filtros. Picos = escena. Valles = resumen.")
    if segs:
        fig, ax = plt.subplots(figsize=(12, 4))
        xs = [s["n"] for s in segs]
        ys = [s["tension"] for s in segs]
        ax.plot(xs, ys, marker="o", color="#c0392b", linewidth=2, markersize=8)
        ax.fill_between(xs, ys, alpha=0.2, color="#c0392b")
        prom = statistics.mean(ys)
        ax.axhline(prom, color="gray", linestyle="--", label=f"Promedio: {prom:.1f}")
        ax.set_xlabel("Segmento →")
        ax.set_ylabel("Tensión")
        ax.set_title("Mapa de tensión narrativa")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        pico = max(ys)
        idx = ys.index(pico) + 1
        if pico > prom * 1.3:
            st.success(f"✅ Hay clímax en el segmento {idx}.")
        else:
            st.error("🔴 **No hay clímax claro.** Di Marco: *'Si la curva es plana, no hay cuento, hay anécdota.'*")
        st.subheader("Valles de tensión (posible resumen)")
        for s in sorted(segs, key=lambda x: x["tension"])[:3]:
            with st.expander(f"Segmento {s['n']} — tensión {s['tension']:.1f}"):
                st.caption(s["preview"])
                st.write(f"Long. prom: {s['long_prom']:.1f} | Diálogo: {'sí' if s['tiene_dialogo'] else 'no'} | Filtros: {s['n_filt']}")

# ---- Tab 2 ----
with tabs[2]:
    st.header("🗡️ Santa Tijera")
    st.caption("Lo que Di Marco llamaría 'rebabas': lo que el lector puede inferir o diluye el efecto.")
    filtros = detectar_filtros(a)
    st.subheader(f"🔍 Palabras filtro ({len(filtros)})")
    st.caption("Di Marco: *'El filtro es una lente sucia. Si escribís \"vio que\", el lector ve el verbo, no la escena.'*")
    if filtros:
        por_cat = defaultdict(list)
        for f in filtros:
            por_cat[f["cat"]].append(f)
        for cat, items in sorted(por_cat.items(), key=lambda x: -len(x[1])):
            with st.expander(f"**{cat}** — {len(items)} apariciones"):
                for it in items[:5]:
                    st.markdown(f"> …{it['contexto']}…")
                if len(items) > 5:
                    st.caption(f"...y {len(items) - 5} más.")
                st.info("💡 **Sugerencia:** Mostrá la acción, no la percepción. En vez de *'vio que la puerta estaba abierta'*, escribí *'la puerta estaba abierta'*.")
    else:
        st.success("✅ Sin filtros detectados.")

    st.divider()
    pasivas = detectar_pasivas(a)
    st.subheader(f"🔍 Voz pasiva ({len(pasivas)})")
    st.caption("La pasiva aleja al lector de la acción.")
    if pasivas:
        for p in pasivas[:10]:
            st.markdown(f"- `{p['frase']}`")
    else:
        st.success("✅ Sin voz pasiva.")

    st.divider()
    cliches = detectar_cliches(texto)
    st.subheader(f"🔍 Clichés ({len(cliches)})")
    if cliches:
        for c in cliches:
            st.warning(f"⚠️ `{c}`")
        st.info("💡 **Di Marco:** El cliché es una imagen que ya no enciende nada. Buscá una imagen propia.")
    else:
        st.success("✅ Sin clichés de la lista.")

    st.divider()
    explicativos = detectar_explicativos(texto)
    st.subheader(f"🔍 Marcadores explicativos ({len(explicativos)})")
    if explicativos:
        for e in explicativos:
            st.warning(f"⚠️ `{e}`")
        st.info("💡 **Di Marco:** Si necesitás explicar, el texto no está haciendo su trabajo. Mostrá y confiá en el lector.")
    else:
        st.success("✅ Sin marcadores.")

# ---- Tab 3 ----
with tabs[3]:
    st.header("🔁 Ecos y rimas no buscadas")
    ecos = detectar_ecos(a)
    st.subheader("🔍 Ecos (palabras que resuenan)")
    st.caption("Di Marco: *'El eco es una rima que no buscaste.'*")
    if ecos:
        for e in ecos[:15]:
            st.warning(f"**'{e['lema']}'** — {e['n']} apariciones")
    else:
        st.success("✅ Sin ecos significativos.")

    st.divider()
    rimas = detectar_rimas(a)
    st.subheader(f"🔍 Rimas no buscadas ({len(rimas)})")
    st.caption("Palabras que terminan igual a corta distancia. En prosa suenan a error.")
    if rimas:
        for r in rimas[:15]:
            st.warning(f"`{r['a']}` / `{r['b']}` (a {r['dist']} palabras)")
    else:
        st.success("✅ Sin rimas accidentales.")

# ---- Tab 4 ----
with tabs[4]:
    st.header("🎭 Efectismo")
    st.caption("Di Marco: *'El efectismo es la sobreabundancia de adornos que hacen que el efecto pierda efecto.'*")
    adjs = detectar_adjetivos(a)
    dens_adj = len(adjs) / max(metricas["n_palabras"], 1)
    st.subheader("Densidad de adjetivos")
    c1, c2 = st.columns(2)
    c1.metric("Adjetivos", len(adjs))
    c2.metric("Densidad", f"{dens_adj * 100:.1f}%")
    if dens_adj > 0.10:
        st.error("🔴 **Densidad alta.** Demasiados adjetivos ahogan al sustantivo. Quitá uno de cada tres.")
    elif dens_adj > 0.07:
        st.warning("🟡 **Densidad media.** Revisá si todos aportan.")
    else:
        st.success("✅ Densidad saludable.")

    st.divider()
    advs = detectar_adverbios_mente(a)
    st.subheader(f"Adverbios en -mente ({len(advs)})")
    st.caption("*'Sonrió alegremente'* es redundante: la sonrisa ya es alegría.")
    if advs:
        for adv in advs[:20]:
            st.warning(f"`{adv.text}`")
    else:
        st.success("✅ Sin adverbios en -mente.")

# ---- Tab 5 ----
with tabs[5]:
    st.header("📝 Reporte completo")
    rep = "# Reporte Di Marco Informático\n\n"
    rep += "## Métricas generales\n\n"
    rep += f"- Palabras: {metricas['n_palabras']}\n"
    rep += f"- Oraciones: {metricas['n_oraciones']}\n"
    rep += f"- Párrafos: {metricas['n_parrafos']}\n"
    rep += f"- Long. promedio oración: {metricas['long_prom']:.1f}\n"
    rep += f"- Long. máxima: {metricas['long_max']}\n"
    rep += f"- Desvío: {metricas['desvio']:.1f}\n"
    rep += f"- Riqueza léxica: {metricas['riqueza'] * 100:.0f}%\n\n"

    rep += "## ¿Hay cuento?\n\n"
    for c in checks:
        icon = "✅" if c["ok"] else "❌"
        rep += f"{icon} **{c['nombre']}** — {c['detalle']}\n"
        rep += f"   > {c['di_marco']}\n\n"

    rep += "## Santa Tijera\n\n"
    rep += f"- Filtros: {len(filtros)}\n"
    rep += f"- Voz pasiva: {len(pasivas)}\n"
    rep += f"- Clichés: {len(cliches)}\n"
    rep += f"- Marcadores explicativos: {len(explicativos)}\n\n"

    rep += "## Ecos y rimas\n\n"
    rep += f"- Ecos: {len(ecos)}\n"
    rep += f"- Rimas: {len(rimas)}\n\n"

    rep += "## Efectismo\n\n"
    rep += f"- Adjetivos: {len(adjs)} ({dens_adj * 100:.1f}%)\n"
    rep += f"- Adverbios -mente: {len(advs)}\n"

    st.markdown(rep)
    st.download_button(
        "📥 Descargar reporte (.md)",
        rep,
        file_name="reporte_dimarco.md",
        mime="text/markdown",
    )

# ---- Tab 6 (Nueva) ----
with tabs[6]:
    st.header("🖼️ Imaginismo (La imagen concreta)")
    st.caption("Di Marco, siguiendo a Ezra Pound, busca la 'imagen concreta y potente'. Lo abstracto debilita; lo sensorial fortalece.")

    abstractos = ["amor", "odio", "tristeza", "alegría", "belleza", "fealdad", "miedo", "valor", "libertad", "justicia", "verdad", "mentira", "paz", "guerra", "dolor", "placer", "soledad", "angustia", "felicidad", "esperanza"]
    sensoriales = ["ver", "mirar", "observar", "oír", "escuchar", "oler", "gustar", "tocar", "sentir", "palpar"]

    hallazgos_abstractos = []
    hallazgos_sensoriales = []

    for token in a["doc"]:
        if token.lemma_.lower() in abstractos:
            start = max(0, token.i - 5)
            end = min(len(a["doc"]), token.i + 6)
            contexto = a["doc"][start:end].text
            hallazgos_abstractos.append({"palabra": token.text, "contexto": contexto})
        
        if token.lemma_.lower() in sensoriales and token.pos_ == "VERB":
            hallazgos_sensoriales.append(token.text)

    st.subheader("Abstracciones detectadas")
    st.caption("Los conceptos abstractos son 'contar'. Si usas 'amor', intenta mostrar el gesto concreto que lo demuestra.")
    if hallazgos_abstractos:
        for h in hallazgos_abstractos[:10]:
            st.warning(f"**'{h['palabra']}'** en contexto: > …{h['contexto']}…")
        if len(hallazgos_abstractos) > 10:
            st.caption(f"...y {len(hallazgos_abstractos) - 10} más.")
    else:
        st.success("✅ Sin abstracciones comunes en el texto. ¡Buen trabajo!")

    st.divider()
    
    st.subheader("Verbos de percepción sensorial")
    st.caption("Un texto con 'imágenes potentes' suele apoyarse en lo que los personajes ven, oyen o tocan.")
    st.write(f"Se detectaron **{len(hallazgos_sensoriales)}** verbos sensoriales.")
    if len(hallazgos_sensoriales) < 3:
        st.warning("🟡 **Pocos anclajes sensoriales.** El texto puede estar flotando en lo abstracto o en lo puramente mental.")
    else:
        st.info("💡 **Di Marco:** 'Va a buscar precisión y belleza, al mismo tiempo'. Asegúrate de que esos verbos estén conectados a una imagen que el lector pueda ver.")

    st.divider()
    
    st.subheader("Reflexión final (Modo Taller)")
    st.info("""
    **El criterio de Di Marco:** 
    "El efecto es lo que nosotros como escritores deseamos transmitir al lector de una manera clara y concisa, sin rebabas; y el efectismo es la sobreabundancia de descripciones... que paradójicamente hacen que el efecto pierda efecto." 
    
    *Pregúntate: ¿Esta imagen concreta acerca al lector al efecto que busco, o solo está adornando?*
    """)
