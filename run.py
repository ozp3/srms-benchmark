"""
v1 — Yalnizca classifier benchmark.
Interaktif menu: model + dataset secimi, calistirma, rapor.

BU DOSYANIN GOREVI:
  Kullaniciya renkli terminal menusu sunar, secimleri alir, ONNX adapter'i
  olusturur, benchmark'i calistirir, sonuclari JSON olarak kaydeder ve HTML
  raporu uretir. Projenin "main" dosyasidir — python run.py ile calistirilir.

AKIS:
  1. _header()   → baslik yaz
  2. _pick()     → model dosyasi sec (models/ icindeki .onnx'ler listelenir)
  3. _pick()     → dataset sec (datasets/ icindeki JSON'lar listelenir)
  4. Ozet + onay → secimleri goster, kullaniciya sor
  5. DistilBERTAdapter olustur
  6. run_classifier_benchmark(adapter, dataset) cagir
  7. Sonuclari results/{timestamp}_classifier/results.json kaydet
  8. report.generate_html() ile HTML raporu olustur
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Bu dosyanin bulundugu klasoru Python path'e ekle — import'larin calismasi icin
_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_dir))

from suites import run_classifier_benchmark
from adapters import DistilBERTAdapter


# =============================================================================
# ANSI RENK KODLARI — terminalde renkli cikti icin
# =============================================================================

def _a(code: str) -> str:
    """
    ANSI escape code uretir. Windows'ta ANSI destegini aktif etmek icin
    os.system("") cagirir (Windows 10+ Terminal'de calisir).
    """
    if sys.platform == "win32":
        os.system("")  # Windows terminalinde ANSI renklerini etkinlestir
    return f"\033[{code}m"

# Renk sabitleri — her yerde bunlari kullan, tutarli olsun
R = _a("0")     # reset — rengi sifirla
B = _a("1")     # bold — kalin yazi
D = _a("2")     # dim — soluk yazi
G  = _a("92")   # green — basarili/onay
Y  = _a("93")   # yellow — uyari/baslik
C  = _a("96")   # cyan — cerceve/menu
W  = _a("97")   # white — vurgu
E  = _a("91")   # red — hata/iptal

def _c(text: str, color: str) -> str:
    """Metni renklendir ve sonunda reset'le."""
    return f"{color}{text}{R}"


# =============================================================================
# MENU YARDIMCILARI — kullanici etkilesimi icin tekrar kullanilan fonksiyonlar
# =============================================================================

def _header():
    """Uygulama basligi — her calistirmada en ustte gorunur."""
    print()
    print(_c("╔══════════════════════════════════════════╗", C))
    print(_c("║", C) + _c("      SRMS-26  AI Benchmark Runner        ", B + W) + _c("║", C))
    print(_c("║", C) + _c("             v1 · classifier only           ", D) + _c("║", C))
    print(_c("╚══════════════════════════════════════════╝", C))
    print()


def _section(title: str):
    """Menu bolumu basligi. Ornek: '1 / 2  —  Model sec'"""
    print()
    print(_c(f"  ┌─ {title} ", Y) + _c("─" * max(0, 42 - len(title)), D))


def _option(idx: int, label: str, sub: str = ""):
    """
    Bir menu secenegi yaz.
    Ornek: '  1)  text_classifier_v10.onnx  ONNX · 255.0 MB · 2026-04-21 19:21'
    """
    print(f"{_c(f'  {idx})', C)}  {_c(label, W)}{_c(f'  {sub}', D) if sub else ''}")


def _ask(msg: str = "") -> str:
    """
    Kullanicidan input al. Ctrl+C veya Ctrl+D ile cikisa izin ver.
    Ornek: '  Secim> 1'
    """
    try:
        return input(_c(f"\n  {msg}> ", G)).strip()
    except (KeyboardInterrupt, EOFError):
        print(_c("\n\n  Iptal edildi.", E))
        sys.exit(0)


def _pick(options: list, title: str) -> str:
    """
    Kullaniciya coktan secmeli menu gosterir ve secim yaptirir.
    Gecerli bir secim yapilana kadar dongude kalir.

    Parametreler:
      options: [(label, value, hint), ...] — her secenek icin gorunen ad, donen deger, aciklama
      title:   menu basligi (section icinde gorunur)

    Return: secilen option'in value'su (ikinci eleman)
    """
    _section(title)
    for i, (label, _, hint) in enumerate(options, 1):
        _option(i, label, hint)
    while True:
        raw = _ask("Secim")
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                chosen = options[idx - 1]
                print(_c(f"  ✓ {chosen[0]}", G))
                return chosen[1]
        print(_c("  Gecersiz secim, tekrar dene.", E))


# =============================================================================
# ANA MENU VE BENCHMARK AKISI
# =============================================================================

def run():
    """Programin ana giris noktasi — interaktif menu + benchmark calistirma."""
    _header()

    # v1'de suite sabit — sadece classifier
    suite = "classifier"

    # ═════════════════════════════════════════════════════════════════════
    # 1. MODEL SECIMI
    # ═════════════════════════════════════════════════════════════════════

    onnx_path = None
    models_dir = _dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # models/ icindeki .onnx dosyalarini tara, en yeni en ustte
    files = sorted(
        [p for p in models_dir.iterdir() if p.suffix == ".onnx"],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )

    if not files:
        print(_c("\n  ⚠  models/ icinde .onnx dosyasi bulunamadi.", E))
    else:
        opts = []
        for f in files:
            # Her dosya icin gorunen bilgi: ad, format, boyut, tarih
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size  = f.stat().st_size / (1024 * 1024)
            opts.append((f.name, str(f), f"ONNX  ·  {size:.1f} MB  ·  {mtime}"))
        onnx_path = _pick(opts, "1 / 2  —  Model sec (.onnx)")

    # ═════════════════════════════════════════════════════════════════════
    # 2. DATASET SECIMI
    # ═════════════════════════════════════════════════════════════════════

    ds_dir = _dir / "datasets"
    ds_opts = []
    txt_path = ds_dir / "text_only.json"

    # Birincil dataset: text_only.json (300 ornek, gorsel gerektirmez)
    if txt_path.exists():
        with open(txt_path, encoding="utf-8") as f:
            n = len(json.load(f))
        ds_opts.append(("text_only.json", "text_only.json", f"{n} ornek · gorsel yok"))
    else:
        ds_opts.append(("text_only.json  ⚠ bulunamadi", "text_only.json", ""))

    # Ek dataset'ler: datasets/ altindaki diger JSON dosyalarini da goster
    # (kullanici kendi test verisini koyabilir)
    for ds_file in sorted(ds_dir.glob("*.json")):
        if ds_file.name == "text_only.json":
            continue
        try:
            with open(ds_file, encoding="utf-8") as f:
                data = json.load(f)
            n = len(data) if isinstance(data, list) else "?"
            ds_opts.append((ds_file.name, ds_file.name, f"{n} ornek"))
        except Exception:
            pass

    dataset = _pick(ds_opts, "2 / 2  —  Dataset sec")

    # ═════════════════════════════════════════════════════════════════════
    # OZET + ONAY
    # ═════════════════════════════════════════════════════════════════════

    print()
    print(_c("  ┌─ Ozet ──────────────────────────────────", C))
    for k, v in [
        ("Suite",    suite),
        ("Dataset",  dataset),
        ("Model",    Path(onnx_path).name if onnx_path else "otomatik"),
    ]:
        print(f"  {_c(k + ':', Y):<24}{_c(v, W)}")
    print(_c("  └────────────────────────────────────────", C))

    confirm = _ask("Calistir? [E/h]")
    if confirm.lower() in ("h", "hayir", "n", "no"):
        print(_c("\n  Iptal edildi.\n", Y))
        sys.exit(0)
    print()

    # ═════════════════════════════════════════════════════════════════════
    # BENCHMARK CALISTIRMA
    # ═════════════════════════════════════════════════════════════════════

    # Dataset'i yukle
    datasets_root = _dir / "datasets"
    ds_path = datasets_root / dataset
    if not ds_path.exists():
        print(f"Dataset bulunamadi: {ds_path}")
        sys.exit(1)
    with open(ds_path, encoding="utf-8") as f:
        ds = json.load(f)

    # ONNX model adapter'ini olustur
    # path belirtilmemisse models/ icindeki en yeni .onnx otomatik secilir
    classifier = DistilBERTAdapter(onnx_path=onnx_path)

    # Sonuc dict'ini hazirla — bu dict results.json'a yazilacak
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(ds_path.name),
        "dataset_size": len(ds),
        "suite": suite,
        "config": {
            "onnx_path": onnx_path or "auto",
        },
        "modules": {},
    }

    # Benchmark'i calistir
    # suites.py'deki run_classifier_benchmark fonksiyonu:
    #   - dataset'teki her ornegi modele sorar
    #   - tum metrikleri hesaplar
    #   - sonuc dict'i dondurur
    print(f"[*] Classifier testi basliyor ({classifier.name})...")
    results["modules"]["classifier"] = run_classifier_benchmark(classifier, ds)
    r = results["modules"]["classifier"]

    # Ozet metrikleri terminale yaz — kullanici hemen gorsun
    print(f"    category_accuracy: {r['category_accuracy']:.1%}")
    print(f"    macro_f1:          {r['category_macro_f1']:.3f}")
    print(f"    priority_exact:    {r['priority_accuracy_exact']:.1%}")
    print(f"    priority_tol1:     {r['priority_accuracy_tol1']:.1%}")
    print(f"    priority_tol2:     {r['priority_accuracy_tol2']:.1%}")
    print(f"    priority_mae:      {r['priority_mae']:.3f}")
    print(f"    ECE:               {r['ece']:.3f}")

    # ═════════════════════════════════════════════════════════════════════
    # SONUCLARI KAYDET
    # ═════════════════════════════════════════════════════════════════════

    # Her calistirma icin timestamp'li benzersiz klasor ac
    # Ornek: results/2026-06-02_143022_classifier/
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _dir / "results" / f"{ts}_{suite}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ham sonuclari JSON olarak kaydet — sonra report.py ile HTML'e cevrilir
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Sonuclar kaydedildi : {json_path}")

    # HTML raporu olustur
    # report.py generate_html() fonksiyonu results dict'ini alip
    # grafikli, tablolu, self-contained HTML dosyasi uretir.
    # Hata olursa benchmark sonucu etkilenmez — sadece uyari basilir.
    try:
        from report import generate_html
        print("[*] HTML raporu olusturuluyor...")
        html = generate_html(results)
        html_path = out_dir / "report.html"
        html_path.write_text(html, encoding="utf-8")
        print(f"[OK] Rapor olusturuldu    : {html_path}")
    except Exception as e:
        print(f"[WARN] HTML raporu atildi : {e}")


if __name__ == "__main__":
    run()
