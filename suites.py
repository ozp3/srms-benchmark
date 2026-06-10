"""
Benchmark test suites — v1: yalnizca classifier.

Bu dosya, classifier modelini test eden ana fonksiyonu icerir.
Dataset uzerinde tek tek dolanir, her ornek icin modeli cagirir,
sonuclari toplar ve metrikleri hesaplar.

Akis:
  1. Dataset'teki her ornek icin adapter.classify(text) cagir
  2. Tahmin edilen kategori, oncelik, birim degerlerini ayri listelere topla
  3. Dongu bitince metrics.py'deki fonksiyonlarla tum metrikleri hesapla
  4. Dict olarak sonuc donder → run.py bunu results.json'a yazar
"""

from adapters import ClassifierAdapter, CLASSES
import metrics as scoring

# Oncelik seviyeleri (string olarak — per_class_prf ile uyumlu olsun diye)
PRIORITY_LEVELS = ["0", "1", "2", "3", "4", "5"]


# =============================================================================
# TEXT CLASSIFIER BENCHMARK
# =============================================================================

def run_classifier_benchmark(
    adapter: ClassifierAdapter,
    dataset: list[dict],
) -> dict:
    """
    Classifier modelini verilen dataset uzerinde test eder.

    Parametreler:
      adapter:  ClassifierAdapter implementasyonu (DistilBERTAdapter veya PyTorchAdapter)
      dataset:  Her biri {"text": "...", "expected_classifier": {...}} formatinda sozluk listesi

    Return dict icerigi:
      - category_accuracy:       kategori dogrulugu (top-1)
      - category_macro_f1:       sinif dengeli F1 ortalamasi
      - category_weighted_f1:    ornek sayisi agirlikli F1 ortalamasi
      - per_class_prf:           her sinif icin precision/recall/F1/support
      - confusion_matrix:        14x14 karmasiklik matrisi
      - priority_accuracy_exact: birebir oncelik eslesmesi (%)
      - priority_accuracy_tol1:  ±1 toleransli oncelik dogrulugu (%)
      - priority_accuracy_tol2:  ±2 toleransli oncelik dogrulugu (%)
      - priority_mae:            oncelik ortalama mutlak hata
      - per_priority_prf:        her oncelik seviyesi icin P/R/F1
      - priority_confusion_matrix: 6x6 oncelik karmasiklik matrisi
      - priority_distribution:   expected vs predicted oncelik dagilimi
      - department_accuracy:     birim esleme dogrulugu
      - ece:                     expected calibration error
      - avg_confidence:          ortalama confidence degeri
      - cases:                   her ornek icin detayli sonuc listesi
    """

    # ── Tahminleri toplamak icin listeler ──────────────────────────────────
    # Her listede ayni index ayni ornege ait. Zip ile eslestirip metrik hesaplanir.

    y_cat_true: list[str] = []   # gercek kategoriler
    y_cat_pred: list[str] = []   # tahmin edilen kategoriler
    y_pri_true: list[int] = []   # gercek oncelikler (0-5)
    y_pri_pred: list[int] = []   # tahmin edilen oncelikler
    y_dep_true: list[str] = []   # gercek birimler (Fen Isleri, Temizlik Isleri...)
    y_dep_pred: list[str] = []   # tahmin edilen birimler

    confidences: list[float] = []  # her tahminin confidence degeri (ECE icin)
    correctness: list[bool] = []   # her tahmin dogru mu? (ECE icin)
    case_results: list[dict] = []  # per-case detay (rapor ve hata analizi icin)

    # ── Ana dongu: her ornegi tek tek test et ─────────────────────────────
    for case in dataset:
        # Dataset sozlugunden beklenen degerleri ve metni al
        expected = case.get("expected_classifier")
        text = case.get("text")
        # expected_classifier veya text yoksa o ornegi atla
        if not expected or not text:
            continue

        # Modeli cagir — iste asil inference burada
        result = adapter.classify(text)

        # Kategori tahminini kaydet
        y_cat_true.append(expected["category"])
        y_cat_pred.append(result["category"])

        # Oncelik varsa kaydet (tum datasetlerde zorunlu)
        if "priority" in expected:
            y_pri_true.append(expected["priority"])
            y_pri_pred.append(result["priority"])

        # Birim varsa kaydet
        if "department" in expected:
            y_dep_true.append(expected["department"])
            y_dep_pred.append(result["department"])

        # Confidence ve dogruluk — ECE hesabi icin gerekli
        confidences.append(result["confidence"])
        correct = result["category"] == expected["category"]
        correctness.append(correct)

        # Per-case detay — raporda "Hatali Tahminler" tablosu buradan gelir
        case_results.append({
            "id": case.get("id"),
            "difficulty": case.get("difficulty"),
            "text": text,
            "expected_category": expected["category"],
            "predicted_category": result["category"],
            "confidence": result["confidence"],
            "correct": correct,
        })

    # ── KATEGORI METRIKLERI ────────────────────────────────────────────────
    cat_acc = scoring.accuracy(y_cat_true, y_cat_pred)
    cat_prf = scoring.per_class_prf(y_cat_true, y_cat_pred, labels=CLASSES)
    cat_cm  = scoring.confusion_matrix(y_cat_true, y_cat_pred, labels=CLASSES)

    # ── ONCELIK METRIKLERI ─────────────────────────────────────────────────
    # Priority degerlerini string'e cevir — per_class_prf string label bekler
    pri_true_str = [str(p) for p in y_pri_true]
    pri_pred_str = [str(p) for p in y_pri_pred]
    per_priority_prf = scoring.per_class_prf(pri_true_str, pri_pred_str, labels=PRIORITY_LEVELS)
    priority_cm      = scoring.confusion_matrix(pri_true_str, pri_pred_str, labels=PRIORITY_LEVELS)

    # Priority dagilimi: model gereksiz yere Critical basiyor mu? diye kontrol
    pri_dist_expected  = {str(i): pri_true_str.count(str(i)) for i in range(6)}
    pri_dist_predicted = {str(i): pri_pred_str.count(str(i)) for i in range(6)}

    # ── SONUC DICT'INI OLUSTUR ─────────────────────────────────────────────
    # Bu dict run.py tarafindan results.json'a yazilir,
    # report.py tarafindan HTML raporuna donusturulur.
    return {
        "module": "classifier",
        "adapter": f"{adapter.name}:{adapter.version}",
        "n_cases": len(y_cat_true),

        # Kategori metrikleri
        "category_accuracy": cat_acc,
        "category_macro_f1": scoring.macro_f1(cat_prf),
        "category_weighted_f1": scoring.weighted_f1(cat_prf),
        "per_class_prf": cat_prf,
        "confusion_matrix": cat_cm,

        # Oncelik metrikleri
        "priority_accuracy_exact": scoring.priority_accuracy(y_pri_true, y_pri_pred, tolerance=0),
        "priority_accuracy_tol1": scoring.priority_accuracy(y_pri_true, y_pri_pred, tolerance=1),
        "priority_accuracy_tol2": scoring.priority_accuracy(y_pri_true, y_pri_pred, tolerance=2),
        "priority_mae": scoring.priority_mae(y_pri_true, y_pri_pred),
        "per_priority_prf": per_priority_prf,
        "priority_confusion_matrix": priority_cm,
        "priority_distribution": {
            "expected":  pri_dist_expected,
            "predicted": pri_dist_predicted,
        },

        # Birim ve guvenilirlik
        "department_accuracy": scoring.accuracy(y_dep_true, y_dep_pred),
        "ece": scoring.expected_calibration_error(confidences, correctness, n_bins=10),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,

        # Per-case detay (rapordaki hata tablosu icin)
        "cases": case_results,
    }
