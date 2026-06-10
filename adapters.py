"""
Model adapters: base abstract class + ONNX classifier implementation.
v1 — yalnizca classifier, yalnizca ONNX.

BU DOSYANIN GOREVI:
  Tek bir arayuz (ClassifierAdapter) arkasinda ONNX Runtime ile DistilBERT
  inference yapar. Disaridan bakan kod (suites.py, run.py) sadece
  ClassifierAdapter.classify(text) metodunu gorur. Bu, Strategy Pattern /
  Adapter Pattern uygulamasidir.

TASARIM KARARLARI:
  1. Neden ABC kullandik? → Ileride baska bir model turu eklenince
     sadece ClassifierAdapter'dan turetip classify()'i implemente etmek yeterli.
     Benchmark kodunda (suites.py) hic degisiklik gerekmez.
  2. Neden ONNX? → Egitilmis PyTorch modeli ONNX'e export edildi. ONNX Runtime
     CPU'da ~2-3x daha hizli inference yapar, torch bagimliligi yoktur, tek bir
     .onnx dosyasi tasimasi kolaydir.
  3. Neden lazy import? → onnxruntime sadece bu adapter kullanildiginda yuklenir.
     Benchmark'in baska parcaciklari bu kutuphaneye bagimli degildir.
"""

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

# transformers'in "PyTorch not found" uyarisini sustur.
# v1'de sadece tokenizer kullaniliyor, PyTorch'a ihtiyac yok.
# Bu env var import'tan ONCE set edilmeli — lazy import sirasinda calisir.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Model dosyalarinin bulundugu klasor — bu dosyanin yanindaki models/
_MODELS_DIR = Path(__file__).resolve().parent / "models"


# =============================================================================
# PAYLASILAN SABITLER
# =============================================================================

# 14 sinifli kategori listesi.
# "normal" = sorun yok ama gercek rapor, "irrelevant" = troll/gecersiz.
# Bu siniflar hem egitimde hem inference'ta ayni sirada olmali.
CLASSES = [
    "road_damage", "sidewalk_damage", "waste", "pollution",
    "green_space", "lighting", "traffic_sign", "sewage_water",
    "infrastructure", "vandalism", "stray_animal", "natural_disaster",
    "normal", "irrelevant",
]

# Oncelik seviyesi → insan okunabilir etiket.
# Model 0-5 arasi integer tahmin eder, bu map ile label'a cevrilir.
PRIORITY_LABELS = {
    0: "Irrelevant", 1: "Normal", 2: "Minor",
    3: "Moderate", 4: "High", 5: "Critical",
}

# Kategori → sorumlu belediye birimi eslemesi.
# Modelin tahmin ettigi kategoriye gore raporun hangi birime gidecegini belirler.
# "normal" ve "irrelevant" islem gormez ("-").
DEPARTMENT_MAP = {
    "road_damage":      "Fen Isleri",
    "sidewalk_damage":  "Fen Isleri",
    "waste":            "Temizlik Isleri",
    "pollution":        "Cevre Koruma",
    "green_space":      "Park ve Bahceler",
    "lighting":         "Elektrik Birimi",
    "traffic_sign":     "Trafik Birimi",
    "sewage_water":     "Su ve Kanalizasyon",
    "infrastructure":   "Fen Isleri",
    "vandalism":        "Zabita",
    "stray_animal":     "Veteriner Birimi",
    "natural_disaster": "Afet Koordinasyon",
    "normal":           "-",
    "irrelevant":       "-",
}

# Model hiperparametreleri.
# CONFIDENCE_THRESHOLD: bu degerin altindaki tahminler "needs_review" isaretlenir
#   → insan operatorun kontrol etmesi istenir. 0.60 = %60 guven esigi.
# MAX_LEN: tokenizer'da maksimum token sayisi. 64 kelime yeterli cunku
#   Gemini'nin description ciktisi max ~15 kelime.
CONFIDENCE_THRESHOLD = 0.60
MODEL_NAME = "distilbert-base-uncased"  # HuggingFace'teki pretrained model adi
MAX_LEN = 64
NUM_CLASSES = len(CLASSES)  # 14
NUM_PRIORITIES = 6          # 0-5


# =============================================================================
# BASE ABSTRACT ADAPTER (sadece classifier)
# =============================================================================

class ClassifierAdapter(ABC):
    """
    Tum classifier adapter'larinin uymasi gereken soyut arayuz (contract).

    Bu siniftan dogrudan nesne olusturulamaz — sadece miras alinir.
    Alt siniflar classify() metodunu implemente etmek zorundadir.

    Neden var: Benchmark kodu (suites.py) sadece ClassifierAdapter tipini bilir.
    Bu sayede yeni bir model eklerken benchmark koduna dokunmayiz — sadece
    yeni bir adapter yazip ClassifierAdapter'dan turetiriz.
    """

    name: str = "unknown_classifier"    # adapter'in insan okunabilir adi
    version: str = "unknown"            # model dosya adi veya versiyon

    @abstractmethod
    def classify(self, text: str) -> dict:
        """
        Verilen metni siniflandirir.

        Bu metod MUTLAKA su alanlari iceren bir dict dondurmelidir:
          - category:       str   (14 siniftan biri)
          - confidence:     float (0.0 - 1.0 arasi softmax olasiligi)
          - priority:       int   (0-5 arasi oncelik seviyesi)
          - priority_label: str   (insan okunabilir oncelik)
          - department:     str   (sorumlu belediye birimi)
          - is_troll:       bool  (kategori "irrelevant" ise True)
          - is_normal:      bool  (kategori "normal" ise True)
          - needs_review:   bool  (confidence < CONFIDENCE_THRESHOLD ise True)
        """
        ...


# =============================================================================
# HELPER FONKSIYONLAR
# =============================================================================

def _find_newest_model(ext: str) -> Path:
    """
    models/ klasorunde en yeni .onnx veya .pth dosyasini bulur.
    Birden fazla varsa degisiklik tarihine (mtime) gore en yeni olani secer.

    Neden var: Kullanici her seferinde --onnx-path vermek zorunda kalmasin.
    models/ klasorune yeni bir model kopyalayinca otomatik algilanir.
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        _MODELS_DIR.glob(f"*{ext}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(
            f"models/ icinde {ext} dosyasi bulunamadi: {_MODELS_DIR}\n"
            f"Modeli oraya kopyala veya path ile acikca belirt."
        )
    if len(files) > 1:
        print(f"[INFO] Birden fazla {ext} bulundu, en yeni secildi: {files[0].name}")
    return files[0]


def _build_classify_output(cls_probs, pri_probs, classes, threshold=CONFIDENCE_THRESHOLD) -> dict:
    """
    Modelin cikti matrislerini (logits → softmax → probs) alip
    ClassifierAdapter.classify() kontratina uygun dict'e cevirir.

    Parametreler:
      cls_probs:  kategori softmax ciktisi — shape (14,) numpy array
      pri_probs:  oncelik softmax ciktisi — shape (6,) numpy array
      classes:    sinif isimleri listesi (CLASSES)
      threshold:  confidence esigi (alti → needs_review=True)

    Neden ayri bir fonksiyon: ONNX adapter ve PyTorch adapter ayni son islemi
    yapiyor. Kod tekrarini onlemek icin ortak fonksiyona cikardik.

    all_scores: Tum siniflarin skorlari (raporda hata analizi icin faydali).
      Ornegin "road_damage tahmin edildi ama 2. sirada waste vardi" gibi.
    """
    # En yuksek olasilikli sinifin index'i
    cls_idx    = int(cls_probs.argmax())
    pri_idx    = int(pri_probs.argmax())
    confidence = float(cls_probs[cls_idx])

    # Tum sinif skorlari — azalan sirada, rapor ve hata analizi icin
    all_scores = sorted(
        [{"class": classes[i], "score": round(float(cls_probs[i]), 4)}
         for i in range(len(classes))],
        key=lambda x: x["score"], reverse=True,
    )

    # Kontrat dict'ini olustur
    return {
        "category":       classes[cls_idx],
        "confidence":     round(confidence, 4),
        "priority":       pri_idx,
        "priority_label": PRIORITY_LABELS.get(pri_idx, "?"),
        "department":     DEPARTMENT_MAP.get(classes[cls_idx], "?"),
        "is_troll":       classes[cls_idx] == "irrelevant",
        "is_normal":      classes[cls_idx] == "normal",
        "needs_review":   confidence < threshold,
        "all_scores":     all_scores,
    }


# =============================================================================
# DISTILBERT ONNX CLASSIFIER ADAPTER
# =============================================================================

class DistilBERTAdapter(ClassifierAdapter):
    """
    ONNX Runtime tabanli DistilBERT classifier.

    ONNX NEDEN: ONNX (Open Neural Network Exchange), egitilmis PyTorch modellerini
    framework bagimsiz formata donusturur. ONNX Runtime ile inference yapmak:
      - PyTorch'tan ~2-3x daha hizlidir (CPU'da)
      - torch bagimliligi yoktur, sadece onnxruntime yeterlidir
      - Model dosyasi tek bir .onnx dosyasidir, tasimasi kolaydir

    AKIS:
      1. Tokenizer (HuggingFace) metni token ID'lere cevirir
      2. ONNX Runtime session'a input_ids + attention_mask verilir
      3. session.run() → 2 adet logits vektoru: class_logits (14) + priority_logits (6)
      4. Softmax ile olasiliga donusturulur
      5. _build_classify_output ile kontrat dict'ine cevrilir
    """

    def __init__(self, onnx_path: str = None):
        """
        ONNX modelini yukler.

        Parametreler:
          onnx_path: .onnx dosyasinin yolu.
                     None → models/ klasorundeki en yeni .onnx otomatik secilir.
        """
        # Lazy import — sadece bu adapter kullanilirsa bu kutuphaneler yuklenir
        from transformers import AutoTokenizer
        import onnxruntime as ort
        import numpy as np

        # Model yolunu belirle
        resolved = Path(onnx_path).resolve() if onnx_path else _find_newest_model(".onnx")
        if not resolved.exists():
            raise FileNotFoundError(f"ONNX dosyasi bulunamadi: {resolved}")

        self.onnx_path = resolved
        self.name      = "distilbert-onnx"
        self.version   = resolved.stem  # dosya adini versiyon olarak kullan

        # ONNX oturumu baslat
        print(f"ONNX model yukleniyor: {resolved}")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._sess      = ort.InferenceSession(
            str(resolved),
            providers=["CPUExecutionProvider"]  # GPU yoksa CPU'da calisir
        )
        self._np = np
        print(f"Model hazir: {resolved.name}\n")

    def classify(self, text: str) -> dict:
        """
        Tek bir metin icin siniflandirma yapar.

        Adimlar:
          1. Tokenize: metin → token ID'leri + attention mask (numpy array)
          2. Inference: ONNX session.run() → class_logits, priority_logits
          3. Softmax:   logits → olasilik dagilimi
          4. Output:    _build_classify_output ile dict'e cevir
        """
        np   = self._np

        # Tokenizasyon — MAX_LEN=64 token, kisa metinler padding ile tamamlanir
        enc  = self._tokenizer(
            text, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="np",
        )
        ids  = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)

        # ONNX inference — iki ayri head'den tahmin alinir
        # class_logits:    14 sinif icin ham skor
        # priority_logits: 6 oncelik seviyesi icin ham skor
        cls_logits, pri_logits = self._sess.run(
            ["class_logits", "priority_logits"],
            {"input_ids": ids, "attention_mask": mask},
        )

        # Softmax: logits → olasilik (0-1 arasi, toplam 1)
        # x.max() cikarmak sayisal kararlilik icin (exp patlamasini onler)
        def softmax(x):
            e = np.exp(x - x.max())
            return e / e.sum()

        return _build_classify_output(
            softmax(cls_logits[0]),  # [0] = batch'teki ilk (ve tek) ornek
            softmax(pri_logits[0]),
            CLASSES,
        )
