"""
Benchmark sonuclarini self-contained HTML raporu olarak gorsellestirir.
v1 — yalnizca classifier raporu.

BU DOSYANIN GOREVI:
  results.json'daki ham metrik verilerini alip; matplotlib grafikleri, CSS ile
  stillendirilmis tablolar ve metric kartlari iceren tek bir HTML dosyasi uretir.
  Grafikler base64 olarak HTML icine gomulur (self-contained) — harici dosya yok.

MIMARI:
  1. _chart_*() fonksiyonlari → matplotlib figuru olusturur, base64 PNG dondurur
  2. _section_classifier() → tum classifier grafiklerini ve tablolarini HTML'e cevirir
  3. generate_html() → ustbilgi + section'lari birlestirip tam HTML dokumani uretir

TASARIM KARARLARI:
  - Neden matplotlib? → Egitim projesi, ek bagimlilik istemedik. plotly daha interaktif
    olurdu ama daha agir. Matplotlib yeterli.
  - Neden base64 inline image? → Tek dosya cikti, email'le paylasimi kolay.
    Harici PNG/JS/CSS yok. Tarayicida direk acilir.
  - Neden karanlik tema? → Modern gorunum, dashboard standarti. Tailwind'in slate
    paletinden esinlenildi.

Kullanim:
  python report.py results/2026-04-22_..._classifier/results.json
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")  # Headless mod — ekran acmaya calisma, dogrudan dosyaya yaz
import matplotlib.pyplot as plt
import numpy as np

from adapters import CLASSES


# =============================================================================
# SABITLER
# =============================================================================

# Sinif isimlerini insan okunabilir formata cevir
# Ornek: "road_damage" → "Road Damage"
CLASS_LABELS = {c: c.replace("_", " ").title() for c in CLASSES}

# Karanlik tema renk paleti (Tailwind slate esintili)
BG_DARK = "#0f172a"   # ana arka plan
BG_CARD = "#1e293b"   # kart/grafik arka plani
BORDER  = "#334155"   # cerceve/ayrac rengi
GREEN   = "#22c55e"   # iyi metrik (>%90)
YELLOW  = "#f59e0b"   # orta metrik (%70-%90)
RED     = "#ef4444"   # kotu metrik (<%70)
PURPLE  = "#a855f7"   # vurgu (adversarial, ozel)
BLUE    = "#3b82f6"   # precision barlari
VIOLET  = "#a78bfa"   # recall barlari
CYAN    = "#22d3ee"   # header cercevesi

# Oncelik seviyesi etiketleri
PRIORITY_LABELS_DISPLAY = {
    "0": "Irrelevant", "1": "Normal", "2": "Minor",
    "3": "Moderate",   "4": "High",   "5": "Critical",
}


# =============================================================================
# YARDIMCI FONKSIYONLAR
# =============================================================================

def _fig_to_b64(fig) -> str:
    """
    matplotlib figurunu base64 PNG string'ine cevirir.
    Bu sayede HTML icine <img src="data:image/png;base64,..."> olarak gomulebilir.
    Islem bitince figuru kapatir (bellek temizligi).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)  # Bellek sizintisini onle — uzun raporlarda onemli
    return b64


def _color(val: float, lo=0.70, hi=0.90) -> str:
    """
    Metrik degerine gore renk kodu dondurur.
    >= hi  (default 0.90) → yesil (iyi)
    >= lo  (default 0.70) → sari (orta)
    <  lo                  → kirmizi (kotu)

    Esik degerleri metrige gore degisebilir. Ornegin priority exact accuracy
    icin daha dusuk esikler kullanilir (cunku birebir oncelik eslesmesi zordur).
    """
    return GREEN if val >= hi else YELLOW if val >= lo else RED


def _styled_fig(w, h):
    """
    Karanlik tema ile stillendirilmis matplotlib figure + axis olusturur.
    Tum grafiklerde tutarli gorunum icin BU fonksiyon kullanilir.
    Direkt plt.subplots() kullanilmaz.
    """
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.yaxis.grid(True, color=BORDER, alpha=0.5)
    ax.set_axisbelow(True)  # grid cizgileri barlarin arkasinda kalsin
    return fig, ax


# =============================================================================
# KATEGORI GRAFIKLERI
# =============================================================================

def _chart_per_class(prf: dict) -> str:
    """
    Sinif bazinda Precision / Recall / F1 grouped bar chart.
    Her sinif icin 3 bar yan yana: mavi=Precision, mor=Recall, renkli=F1.
    F1 bari yesil/sari/kirmizi renklenir (performansa gore).

    Bu grafik sayesinde tek bakista hangi sinifta modelin zayif oldugu gorulur.
    """
    classes = list(prf.keys())
    labels  = [CLASS_LABELS.get(c, c) for c in classes]
    f1s     = [prf[c]["f1"]       for c in classes]
    precs   = [prf[c]["precision"] for c in classes]
    recs    = [prf[c]["recall"]    for c in classes]
    colors  = [_color(f) for f in f1s]
    x, w    = np.arange(len(classes)), 0.27

    fig, ax = _styled_fig(12, 5)
    ax.bar(x - w, precs, w, label="Precision", color=BLUE,   alpha=0.85)
    ax.bar(x,     recs,  w, label="Recall",    color=VIOLET, alpha=0.85)
    bars = ax.bar(x + w, f1s, w, label="F1", color=colors, alpha=0.95)

    for bar, f in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{f:.2f}", ha="center", va="bottom", color="white", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", color="white", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="white")
    ax.set_title("Per-Class  Precision / Recall / F1", color="white", fontsize=13, pad=14)
    ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor="white")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_confusion(cm: dict) -> str:
    """
    14x14 karmasiklik matrisi heatmap'i. Her hucrede o sinifa ait ornek sayisi
    yazar. Kosegen = dogru tahminler (mavi yogunluklu), kosegen disi = hatalar.

    Satir normalize edilir — her satirin toplami 1.0 olur. Bu sayede sinif
    buyuklugunden bagimsiz olarak karistirma orani gorulur.
    """
    classes = list(cm.keys())
    labels  = [CLASS_LABELS.get(c, c) for c in classes]
    n       = len(classes)
    mat     = np.array([[cm[tc].get(pc, 0) for pc in classes] for tc in classes], dtype=float)
    sums    = mat.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1  # Sifira bolmeyi onle (bos siniflar)
    norm    = mat / sums

    fig, ax = _styled_fig(12, 10)
    im      = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    cbar    = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", color="white", fontsize=8)
    ax.set_yticklabels(labels, color="white", fontsize=8)
    ax.set_xlabel("Predicted", color="white", fontsize=11)
    ax.set_ylabel("True",      color="white", fontsize=11)
    ax.set_title("Confusion Matrix  (row-normalized)", color="white", fontsize=13, pad=14)

    for i in range(n):
        for j in range(n):
            v = int(mat[i, j])
            if v > 0:
                # Koyu mavi hucrede siyah yazi, acik mavi hucrede beyaz yazi
                tc = "black" if norm[i, j] > 0.55 else "white"
                ax.text(j, i, str(v), ha="center", va="center",
                        color=tc, fontsize=8,
                        fontweight="bold" if i == j else "normal")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_difficulty(cases: list) -> str:
    """
    Zorluk seviyesine gore accuracy bar chart.
    4 bar: Easy (yesil), Medium (sari), Hard (kirmizi), Adversarial (mor).
    Her barin uzerinde dogru/toplam ve yuzde yazar.

    Bu grafik, modelin hangi zorluk seviyesinde takildigini gosterir.
    Adversarial accuracy dusukse → troll filtre ihtiyaci var demektir.
    """
    diffs  = ["easy", "medium", "hard", "adversarial"]
    lbls   = ["Easy", "Medium", "Hard", "Adversarial"]
    cols   = [GREEN, YELLOW, RED, PURPLE]
    rows   = []
    for d, lbl, col in zip(diffs, lbls, cols):
        grp = [c for c in cases if c.get("difficulty") == d]
        if grp:
            ok = sum(1 for c in grp if c["correct"])
            rows.append((lbl, ok, len(grp), ok / len(grp), col))

    fig, ax = _styled_fig(7, 4)
    for i, (lbl, ok, total, acc, col) in enumerate(rows):
        ax.bar(i, acc, color=col, alpha=0.85, width=0.5)
        ax.text(i, acc + 0.025, f"{ok}/{total}\n{acc:.0%}",
                ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], color="white", fontsize=11)
    ax.set_ylim(0, 1.28)
    ax.set_ylabel("Accuracy", color="white")
    ax.set_title("Accuracy by Difficulty", color="white", fontsize=13, pad=14)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_confidence(cases: list) -> str:
    """
    Confidence dagilimi histogrami. Dogru tahminler yesil, yanlislar kirmizi.
    Kesik sari cizgi = CONFIDENCE_THRESHOLD (0.60).

    Bu grafik su sorulara cevap verir:
    - Yanlis tahminlerin confidence degeri yuksek mi? (asiri ozguven sorunu)
    - 0.60 esigi dogru yerde mi? Cok fazla dogru tahmin esigin altinda kaliyorsa
      esik dusurulmeli, cok fazla yanlis tahmin esigin ustundeyse yukseltilmeli.
    """
    ok_c  = [c["confidence"] for c in cases if     c["correct"]]
    err_c = [c["confidence"] for c in cases if not c["correct"]]
    bins  = np.linspace(0, 1, 21)

    fig, ax = _styled_fig(8, 4)
    ax.hist(ok_c,  bins=bins, alpha=0.75, color=GREEN, label=f"Correct ({len(ok_c)})")
    ax.hist(err_c, bins=bins, alpha=0.75, color=RED,   label=f"Wrong ({len(err_c)})")
    ax.axvline(0.60, color=YELLOW, linestyle="--", linewidth=1.5, label="Threshold (0.60)")
    ax.set_xlabel("Confidence", color="white")
    ax.set_ylabel("Count",      color="white")
    ax.set_title("Confidence Distribution", color="white", fontsize=13, pad=14)
    ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor="white")
    plt.tight_layout()
    return _fig_to_b64(fig)


# =============================================================================
# PRIORITY (ONCELIK) GRAFIKLERI
# =============================================================================

def _chart_priority_confusion(cm: dict) -> str:
    """
    6x6 oncelik karmasiklik matrisi heatmap'i. Kosegende ideal durum: model
    her oncelik seviyesini dogru tahmin ediyor. Kosegenden uzaklasma = ciddi hata.

    Ornek: Gercek oncelik 5 (Critical), tahmin 0 (Irrelevant) → en tehlikeli hata.
    Bu matriste kosegenden uzak hucreler boyle kritik hatalari gosterir.
    """
    levels = ["0", "1", "2", "3", "4", "5"]
    labels = [f"{k}\n{PRIORITY_LABELS_DISPLAY[k]}" for k in levels]
    n      = len(levels)
    mat    = np.array([[cm.get(t, {}).get(p, 0) for p in levels] for t in levels], dtype=float)
    sums   = mat.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    norm   = mat / sums

    fig, ax = _styled_fig(8, 6)
    im   = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, color="white", fontsize=8)
    ax.set_yticklabels(labels, color="white", fontsize=8)
    ax.set_xlabel("Predicted Priority", color="white", fontsize=11)
    ax.set_ylabel("True Priority",      color="white", fontsize=11)
    ax.set_title("Priority Confusion Matrix  (row-normalized)", color="white", fontsize=13, pad=14)

    for i in range(n):
        for j in range(n):
            v = int(mat[i, j])
            if v > 0:
                tc = "black" if norm[i, j] > 0.55 else "white"
                ax.text(j, i, str(v), ha="center", va="center",
                        color=tc, fontsize=10,
                        fontweight="bold" if i == j else "normal")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_per_priority_prf(prf: dict) -> str:
    """
    Her oncelik seviyesi (0-5) icin Precision / Recall / F1 grouped bar chart.
    Kategori grafigiyle ayni yapi, sadece siniflar priority seviyeleri.

    Bu grafik, modelin hangi oncelik seviyesinde tutarli oldugunu gosterir.
    Ornegin "Critical (5)" icin recall dusukse → model kritik raporlari kaciriyor.
    """
    levels = ["0", "1", "2", "3", "4", "5"]
    labels = [f"{k} ({PRIORITY_LABELS_DISPLAY[k]})" for k in levels]
    precs  = [prf.get(k, {}).get("precision", 0) for k in levels]
    recs   = [prf.get(k, {}).get("recall",    0) for k in levels]
    f1s    = [prf.get(k, {}).get("f1",        0) for k in levels]
    colors = [_color(f) for f in f1s]
    x, w   = np.arange(len(levels)), 0.27

    fig, ax = _styled_fig(10, 4)
    ax.bar(x - w, precs, w, label="Precision", color=BLUE,   alpha=0.85)
    ax.bar(x,     recs,  w, label="Recall",    color=VIOLET, alpha=0.85)
    bars = ax.bar(x + w, f1s, w, label="F1", color=colors, alpha=0.95)

    for bar, f in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{f:.2f}", ha="center", va="bottom", color="white", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", color="white", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", color="white")
    ax.set_title("Per-Priority  Precision / Recall / F1", color="white", fontsize=13, pad=14)
    ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor="white")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_priority_distribution(dist: dict) -> str:
    """
    Expected vs Predicted oncelik dagilimi grouped bar chart.
    Sol bar = dataset'teki gercek dagilim, sag bar = modelin tahmin dagilimi.

    Bu grafik su soruyu cevaplar: Model gereksiz yere yuksek oncelik mi veriyor?
    Ornegin "Critical" predicted >> expected ise model alarmist davraniyor demektir.
    Tam tersi ise kritik raporlari gozden kaciriyor.
    """
    levels  = ["0", "1", "2", "3", "4", "5"]
    labels  = [f"{k}\n{PRIORITY_LABELS_DISPLAY[k]}" for k in levels]
    exp_v   = [dist.get("expected",  {}).get(k, 0) for k in levels]
    pred_v  = [dist.get("predicted", {}).get(k, 0) for k in levels]
    x, w    = np.arange(len(levels)), 0.35

    fig, ax = _styled_fig(9, 4)
    ax.bar(x - w / 2, exp_v,  w, label="Expected",  color=BLUE,   alpha=0.85)
    ax.bar(x + w / 2, pred_v, w, label="Predicted", color=VIOLET, alpha=0.85)

    for i, (e, p) in enumerate(zip(exp_v, pred_v)):
        if e: ax.text(i - w / 2, e + 0.3, str(e), ha="center", color="white", fontsize=9)
        if p: ax.text(i + w / 2, p + 0.3, str(p), ha="center", color="white", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.set_ylabel("Count", color="white")
    ax.set_title("Priority Distribution  —  Expected vs Predicted", color="white", fontsize=13, pad=14)
    ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor="white")
    plt.tight_layout()
    return _fig_to_b64(fig)


# =============================================================================
# HTML BILESENLERI
# =============================================================================

def _img(b64: str, alt: str = "") -> str:
    """Base64 PNG verisini <img> tag'ine cevirir. Self-contained HTML'in sirri bu."""
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def _card(label: str, value: str, sub: str = "", color: str = "white") -> str:
    """
    Metrik karti HTML'i. Ust kisimda label (kucuk, gri), ortada buyuk deger,
    altta aciklama. Renk parametresi ile degerin rengi ayarlanir.

    Ornek cikti:
      ┌──────────────────┐
      │ CATEGORY ACCURACY │
      │      90.7%        │  ← color=GREEN
      │  272/300 dogru    │
      └──────────────────┘
    """
    return (f'<div class="card">'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-value" style="color:{color}">{value}</div>'
            f'<div class="card-sub">{sub}</div>'
            f'</div>')


def _section_classifier(cls: dict) -> str:
    """
    Classifier sonuclarini HTML section olarak uretir.
    Icerik: ozet kartlar, zorluk seviyesi, 5 grafik, 3 tablo.

    Bu fonksiyon generate_html() tarafindan cagrilir.
    cls dict'i = results["modules"]["classifier"].
    """
    cases    = cls.get("cases", [])
    failed   = [c for c in cases if not c["correct"]]
    n        = cls["n_cases"]
    cat_acc  = cls["category_accuracy"]
    mf1      = cls["category_macro_f1"]
    pri_ex   = cls["priority_accuracy_exact"]
    pri_tol  = cls["priority_accuracy_tol1"]
    ece      = cls["ece"]
    avg_conf = cls["avg_confidence"]

    # Grafikleri olustur (terminalde ilerleme gostergesi)
    print("  [classifier] Per-class grafik...")
    g_cls  = _chart_per_class(cls["per_class_prf"])
    print("  [classifier] Confusion matrix...")
    g_cm   = _chart_confusion(cls["confusion_matrix"])
    print("  [classifier] Difficulty accuracy...")
    g_diff = _chart_difficulty(cases)
    print("  [classifier] Confidence distribution...")
    g_conf = _chart_confidence(cases)

    # Priority grafikleri (opsiyonel — veri varsa olusturulur)
    pri_mae  = cls.get("priority_mae")
    pri_tol2 = cls.get("priority_accuracy_tol2")
    per_pri_prf = cls.get("per_priority_prf", {})
    pri_cm      = cls.get("priority_confusion_matrix", {})
    pri_dist    = cls.get("priority_distribution", {})

    g_pri_prf  = _chart_per_priority_prf(per_pri_prf)  if per_pri_prf else None
    g_pri_cm   = _chart_priority_confusion(pri_cm)     if pri_cm      else None
    g_pri_dist = _chart_priority_distribution(pri_dist) if pri_dist    else None

    # ── Zorluk seviyesi kartlari ──────────────────────────────────────────
    def diff_row(d, lbl):
        """Tek bir zorluk seviyesi icin HTML kart."""
        grp = [c for c in cases if c.get("difficulty") == d]
        if not grp:
            return ""
        ok  = sum(1 for c in grp if c["correct"])
        acc = ok / len(grp)
        col = _color(acc)
        return (f'<div class="diff-card">'
                f'<div class="label">{lbl}</div>'
                f'<div class="value" style="color:{col}">{ok}/{len(grp)} ({acc:.0%})</div>'
                f'</div>')

    diff_cards = (diff_row("easy", "Easy") + diff_row("medium", "Medium") +
                  diff_row("hard", "Hard") + diff_row("adversarial", "Adversarial"))

    # ── Per-class tablo satirlari ─────────────────────────────────────────
    per_class_rows = ""
    for cn, m in cls["per_class_prf"].items():
        col = _color(m["f1"])
        per_class_rows += (
            f"<tr><td>{CLASS_LABELS.get(cn, cn)}</td>"
            f"<td>{m['precision']:.3f}</td><td>{m['recall']:.3f}</td>"
            f"<td style='color:{col};font-weight:700'>{m['f1']:.3f}</td>"
            f"<td class='muted'>{m['support']}</td></tr>"
        )

    # ── Hatali tahminler tablosu ──────────────────────────────────────────
    failed_rows = ""
    for c in failed:
        # Confidence > 0.6 ama yanlis → sari (asiri ozguven)
        # Confidence < 0.6 ve yanlis → kirmizi (zaten emin degildi)
        cc = YELLOW if c["confidence"] > 0.6 else RED
        failed_rows += (
            f"<tr><td><code>{c['id']}</code></td>"
            f"<td><span class='badge badge-{c.get('difficulty','')}'>{c.get('difficulty','')}</span></td>"
            f"<td class='muted'>{c['text']}</td>"
            f"<td class='ok'>{c['expected_category']}</td>"
            f"<td class='fail'>{c['predicted_category']}</td>"
            f"<td style='color:{cc}'>{c['confidence']:.3f}</td></tr>"
        )

    # ── Priority per-seviye tablo satirlari ───────────────────────────────
    pri_prf_rows = ""
    for lvl in ["0", "1", "2", "3", "4", "5"]:
        m    = per_pri_prf.get(lvl, {})
        col  = _color(m.get("f1", 0))
        exp_count  = pri_dist.get("expected",  {}).get(lvl, 0)
        pred_count = pri_dist.get("predicted", {}).get(lvl, 0)
        pri_prf_rows += (
            f"<tr><td>{lvl} — {PRIORITY_LABELS_DISPLAY.get(lvl, lvl)}</td>"
            f"<td>{m.get('precision', 0):.3f}</td>"
            f"<td>{m.get('recall',    0):.3f}</td>"
            f"<td style='color:{col};font-weight:700'>{m.get('f1', 0):.3f}</td>"
            f"<td class='muted'>{exp_count}</td>"
            f"<td class='muted'>{pred_count}</td></tr>"
        )

    # Priority MAE ve ±2 kartlari (varsa)
    mae_card  = _card("Priority MAE", f"{pri_mae:.3f}",  "ortalama sapma ↓ iyi", _color(1 - min(pri_mae / 3, 1))) if pri_mae is not None else ""
    tol2_card = _card("Priority ±2",  f"{pri_tol2:.1%}", "genis tolerans",        _color(pri_tol2)) if pri_tol2 is not None else ""

    # Priority grafikleri HTML — veriye gore esnek layout
    priority_charts = ""
    if g_pri_prf:
        priority_charts += f'<div class="chart-full"><div class="chart-box">{_img(g_pri_prf, "Per-Priority PRF")}</div></div>'
    if g_pri_dist and g_pri_cm:
        priority_charts += (f'<div class="chart-grid">'
                            f'<div class="chart-box">{_img(g_pri_dist, "Priority Distribution")}</div>'
                            f'<div class="chart-box">{_img(g_pri_cm,   "Priority Confusion Matrix")}</div>'
                            f'</div>')
    elif g_pri_dist:
        priority_charts += f'<div class="chart-full"><div class="chart-box">{_img(g_pri_dist, "Priority Distribution")}</div></div>'
    elif g_pri_cm:
        priority_charts += f'<div class="chart-full"><div class="chart-box">{_img(g_pri_cm, "Priority Confusion Matrix")}</div></div>'

    # ── Tum section HTML'ini birlestir ─────────────────────────────────────
    return f"""
<div class="module-header">Classifier &nbsp;<span class="chip">{cls.get('adapter','distilbert-onnx')}</span></div>

<h2>Ozet Metrikler</h2>
<div class="cards">
  {_card("Category Accuracy", f"{cat_acc:.1%}", f"{int(cat_acc*n)}/{n} dogru", _color(cat_acc))}
  {_card("Macro F1",          f"{mf1:.3f}",    "sinif dengeli",               _color(mf1))}
  {_card("Priority Exact",    f"{pri_ex:.1%}", "birebir eslesme",             _color(pri_ex, 0.5, 0.7))}
  {_card("Priority ±1",       f"{pri_tol:.1%}","toleransli",                  _color(pri_tol))}
  {mae_card}
  {tol2_card}
  {_card("ECE",               f"{ece:.3f}",    "kalibrasyon ↓ iyi",           _color(1-ece))}
  {_card("Avg Confidence",    f"{avg_conf:.3f}","ortalama guven",              _color(avg_conf))}
</div>

<h2>Zorluk Seviyesine Gore</h2>
<div class="diff-grid">{diff_cards}</div>

<h2>Kategori Grafikleri</h2>
<div class="chart-full"><div class="chart-box">{_img(g_cls, "Per-Class F1")}</div></div>
<div class="chart-grid">
  <div class="chart-box">{_img(g_diff, "Difficulty Accuracy")}</div>
  <div class="chart-box">{_img(g_conf, "Confidence Distribution")}</div>
</div>
<div class="chart-full"><div class="chart-box">{_img(g_cm, "Confusion Matrix")}</div></div>

<h2>Per-Class Detay</h2>
<div class="table-wrap">
  <table>
    <thead><tr><th>Kategori</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
    <tbody>{per_class_rows}</tbody>
  </table>
</div>

<h2>Priority Grafikleri</h2>
{priority_charts}

<h2>Priority Detay (Per-Level)</h2>
<div class="table-wrap">
  <table>
    <thead><tr><th>Seviye</th><th>Precision</th><th>Recall</th><th>F1</th><th>Expected</th><th>Predicted</th></tr></thead>
    <tbody>{pri_prf_rows}</tbody>
  </table>
</div>

<h2>Hatali Tahminler &nbsp;({len(failed)} / {n})</h2>
<div class="table-wrap">
  <table>
    <thead><tr><th>ID</th><th>Zorluk</th><th>Metin</th><th>Beklenen</th><th>Tahmin</th><th>Confidence</th></tr></thead>
    <tbody>{failed_rows}</tbody>
  </table>
</div>
"""


# =============================================================================
# ANA HTML SABLONU
# =============================================================================

# Self-contained CSS — harici dosya yok. Karanlik tema, responsive.
# Tum stillendirme burada, tarayici direk acar.
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f172a; color: #e2e8f0; line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; padding: 36px 24px; }
h1  { font-size: 26px; font-weight: 700; color: #f8fafc; }
h2  { font-size: 12px; font-weight: 600; color: #475569; text-transform: uppercase;
      letter-spacing: 0.08em; margin: 40px 0 12px;
      border-bottom: 1px solid #1e293b; padding-bottom: 8px; }
.meta { color: #64748b; font-size: 13px; margin-top: 6px; }
.module-header { font-size: 20px; font-weight: 700; color: #f1f5f9;
                 margin: 52px 0 4px; padding-top: 24px;
                 border-top: 2px solid #334155; }
.chip { display: inline-block; background: #1e293b; border: 1px solid #334155;
        border-radius: 6px; padding: 2px 10px; font-size: 13px;
        font-weight: 500; color: #94a3b8; vertical-align: middle; margin-left: 6px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr));
         gap: 14px; margin: 16px 0; }
.card { background: #1e293b; border: 1px solid #334155; border-radius: 12px;
        padding: 18px 14px; text-align: center; }
.card-label { font-size: 11px; color: #64748b; text-transform: uppercase;
              letter-spacing: 0.08em; margin-bottom: 8px; }
.card-value { font-size: 28px; font-weight: 700; }
.card-sub   { font-size: 11px; color: #64748b; margin-top: 4px; }
.diff-grid  { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 14px 0; }
.diff-card  { background: #1e293b; border: 1px solid #334155; border-radius: 10px;
              padding: 14px; text-align: center; }
.diff-card .label { font-size: 11px; color: #64748b; text-transform: uppercase;
                    letter-spacing: 0.06em; margin-bottom: 6px; }
.diff-card .value { font-size: 20px; font-weight: 700; }
.chart-full { margin: 16px 0; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }
.chart-box  { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px; }
.chart-box img { width: 100%; border-radius: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { text-align: left; padding: 10px 14px; border-bottom: 2px solid #334155;
     color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
td { padding: 10px 14px; border-bottom: 1px solid #1e293b; vertical-align: top; }
tr:hover td { background: #1e293b; }
.table-wrap { background: #0f172a; border: 1px solid #334155; border-radius: 12px;
              overflow: hidden; margin: 12px 0; }
.muted { color: #64748b; font-size: 13px; }
.ok    { color: #22c55e; font-weight: 600; }
.fail  { color: #ef4444; font-weight: 600; }
code   { background: #1e293b; padding: 2px 6px; border-radius: 4px;
         font-size: 12px; color: #7dd3fc; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 9999px;
         font-size: 11px; font-weight: 600; text-transform: uppercase; }
.badge-easy        { background:#14532d; color:#86efac; }
.badge-medium      { background:#713f12; color:#fde68a; }
.badge-hard        { background:#7f1d1d; color:#fca5a5; }
.badge-adversarial { background:#4a1d96; color:#ddd6fe; }
@media (max-width:768px) {
  .chart-grid { grid-template-columns: 1fr; }
  .diff-grid  { grid-template-columns: repeat(2,1fr); }
}
"""


def generate_html(data: dict) -> str:
    """
    results.json dict'ini alip tam HTML dokumani olarak dondurur.

    Parametre: data = JSON.parse("results.json") sonucu.
    Return:    <!DOCTYPE html>... seklinde komple HTML string.

    run.py bu fonksiyonu benchmark bittikten sonra cagirir,
    ciktiyi report.html olarak kaydeder.
    """
    modules  = data.get("modules", {})
    suite    = data.get("suite", "unknown")
    dataset  = data.get("dataset", "—")
    n_cases  = data.get("dataset_size", "?")
    ts       = data.get("timestamp", "")[:19].replace("T", " ")

    # Sadece classifier section'i — v1'de API ve pipeline yok
    body = ""
    if "classifier" in modules:
        body += _section_classifier(modules["classifier"])

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>SRMS-26 Benchmark Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>SRMS-26 &nbsp;·&nbsp; Benchmark Report &nbsp;<span style="font-size:14px;color:#64748b">v1</span></h1>
  <p class="meta">
    Suite: <strong>{suite}</strong>
    &nbsp;|&nbsp; Dataset: <strong>{dataset}</strong>
    &nbsp;|&nbsp; {n_cases} case
    &nbsp;|&nbsp; {ts}
  </p>
  {body}
</div>
</body>
</html>"""


# =============================================================================
# CLI — bagimsiz rapor olusturma
# =============================================================================

def main():
    """
    Komut satirindan bagimsiz rapor olusturma.
    Kullanim: python report.py results/2026-06-02_..._classifier/results.json

    Bu sayede benchmark'i tekrar calistirmadan, kaydedilmis bir results.json'dan
    HTML raporu yeniden uretilebilir.
    """
    parser = argparse.ArgumentParser(description="Benchmark JSON → HTML raporu")
    parser.add_argument("result",   help="Sonuc JSON dosyasi")
    parser.add_argument("--output", default=None, help="Cikti HTML (varsayilan: ayni klasor)")
    args = parser.parse_args()

    path = Path(args.result)
    if not path.exists():
        print(f"Dosya bulunamadi: {path}")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("modules"):
        print("Sonuc dosyasinda modul verisi bulunamadi.")
        sys.exit(1)

    print(f"Rapor olusturuluyor: {path.name}")
    html = generate_html(data)

    out = Path(args.output) if args.output else path.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
