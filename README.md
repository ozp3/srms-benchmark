# SRMS-26 AI Benchmark - v1

**Classifier-only benchmark. ONNX modeli, JSON + HTML rapor ciktisi.**

v1, orijinal benchmark projesinin sadelestirilmis versiyonudur. API cagrilari ve pipeline testleri cikarilmis, sadece text classifier testi birakilmistir. Amac: code review'da odaklanacak yuzeyi kucultmek.

---

## Dosya Yapisi

```
v1/
├── adapters.py        Model yukleme ve inference
├── suites.py          Benchmark dongusu
├── metrics.py         Metrik hesaplama
├── run.py             Ana calistirici + menu
├── report.py          HTML raporu uretme
├── datasets/
│   ├── text_only.json          300 ornek (easy/medium/hard/adversarial)
│   └── text_only.example.json   10 ornek
├── models/
│   └── text_classifier_v10.onnx  256 MB ONNX modeli
└── results/            Her calistirma icin timestamp'li klasor
    └── .../results.json + report.html
```

## Her Dosyanin Gorevi

### adapters.py (289 satir)

Model adapter'lerini tanimlar. **Strategy Pattern** kullanir: dis dunya sadece `ClassifierAdapter` arayuzunu gorur, altinda ONNX mi PyTorch mu olduguyla ilgilenmez (v1'de sadece ONNX vardir).

| Sinif / Fonksiyon | Gorev |
|---|---|
| `ClassifierAdapter` | Soyut sinif (ABC). Tum adapter'ler `classify(text)` metodunu implemente etmek zorunda |
| `DistilBERTAdapter` | ONNX Runtime ile DistilBERT inference. `classify(text)` tek bir metni 14 sinifa ayirir, 0-5 oncelik verir |
| `_find_newest_model()` | `models/` icindeki en yeni `.onnx` dosyasini bulur |
| `_build_classify_output()` | Softmax ciktisini benchmark kontratina uygun dict'e cevirir |

**Sabitler:** `CLASSES` (14 sinif), `PRIORITY_LABELS` (0-5 etiket), `DEPARTMENT_MAP` (kategori -> belediye birimi), `CONFIDENCE_THRESHOLD` (0.60).

### suites.py (162 satir)

`run_classifier_benchmark(adapter, dataset)` fonksiyonu:

1. Dataset'teki her ornek icin `adapter.classify(text)` cagirir
2. Kategori, oncelik, birim tahminlerini listelere toplar
3. Dongu bitince `metrics.py`'deki fonksiyonlarla tum metrikleri hesaplar
4. Dict olarak sonuc dondurur

### metrics.py (192 satir)

Dis bagimliligi olmayan metrik fonksiyonlari (sklearn kullanilmaz).

| Fonksiyon | Olctugu Sey |
|---|---|
| `accuracy()` | Dogru tahmin / toplam ornek |
| `confusion_matrix()` | 14x14 karmasiklik matrisi |
| `per_class_prf()` | Her sinif icin precision, recall, F1, support |
| `macro_f1()` | Sinif dengeli F1 ortalamasi |
| `weighted_f1()` | Ornek sayisi agirlikli F1 ortalamasi |
| `priority_accuracy()` | Oncelik dogrulugu (tolerans: 0, 1, 2) |
| `priority_mae()` | Oncelik ortalama mutlak hata |
| `expected_calibration_error()` | ECE: confidence ile gercek dogruluk arasindaki fark |

### run.py (294 satir)

Programin giris noktasi. `python run.py` ile calisir.

**Akis:**
1. Renkli terminal menusu gosterir
2. Kullaniciya model dosyasi sectirir (`models/` icindeki `.onnx`'ler)
3. Kullaniciya dataset sectirir (`datasets/` icindeki JSON'lar)
4. Ozet gosterir, onay alir
5. `DistilBERTAdapter` olusturur
6. `run_classifier_benchmark()` cagirir
7. Sonuclari `results/{timestamp}_classifier/results.json` kaydeder
8. `report.py` ile HTML raporu olusturur

### report.py (719 satir)

`results.json`'daki verileri self-contained HTML raporuna cevirir. Grafikler matplotlib ile cizilir, base64 olarak HTML icine gomulur. Harici dosya yoktur, tarayiciyla direk acilir.

**Icerik:** Ozet metrik kartlari, per-class F1 bar chart, confusion matrix heatmap, zorluk seviyesine gore accuracy, confidence distribution histogram, priority grafikleri, hatali tahminler tablosu.

## Nasil Calistirilir

```powershell
# v1 klasorune gir
cd v1

# Interaktif menu ile calistir (onerilen)
python run.py

# Belli bir ONNX modeli ile
python run.py --onnx-path "models\text_classifier_v10.onnx"
```

Menu adimlari:
```
1 / 2  --  Model sec (.onnx)
  1)  text_classifier_v10.onnx  ONNX  -  255.0 MB  -  2026-06-02 14:34
  Secim> 1

2 / 2  --  Dataset sec
  1)  text_only.json  300 ornek - gorsel yok
  Secim> 1

Ozet
  Suite:    classifier
  Dataset:  text_only.json
  Model:    text_classifier_v10.onnx

Calistir? [E/h]
```

Cikti ornegi:
```
[*] Classifier testi basliyor (distilbert-onnx)...
    category_accuracy: 90.7%
    macro_f1:          0.905
    priority_exact:    62.7%
    priority_tol1:     94.0%
    priority_tol2:     99.3%
    priority_mae:      0.443
    ECE:               0.068

[OK] Sonuclar kaydedildi : results/2026-06-02_193922_classifier/results.json
[OK] Rapor olusturuldu    : results/2026-06-02_193922_classifier/report.html
```

## Dataset: text_only.json

300 adet elle yazilmis metin ornegi. Metin formati Gemini API'nin description ciktisini taklit eder: `[problem] + [state] + [location]`, Ingilizce, max 15 kelime, artikelsiz.

```json
{
  "id": "TXT_ROAD_E001",
  "text": "pothole cracking asphalt lane near busy intersection",
  "difficulty": "easy",
  "expected_classifier": {
    "category": "road_damage",
    "priority": 4,
    "department": "Fen Isleri"
  }
}
```

| Zorluk | Adet | Icerik |
|---|---|---|
| easy | 105 | Net tek kategori, guclu keyword |
| medium | 90 | Coklu sorun, parafraz, oncelik siniri |
| hard | 60 | Cok kisa metin, yazim hatali, yaniltici baslangic |
| adversarial | 45 | Gemini troll ciktisi simulasyonu (selfie, ic mekan, cizim) |

## Metrikler

### Kategori (14 sinif)

- **Category Accuracy**: Dogru tahmin edilenlerin orani. %90.7 = 300 ornekte 272 dogru.
- **Macro F1**: Her sinifa esit agirlik veren F1 ortalamasi. Sinif dengesizliginden etkilenmez.
- **Weighted F1**: Sinif basina ornek sayisiyla agirlikli F1. Gercek dagilima daha yakin.
- **Confusion Matrix**: Hangi sinifin hangi sinifla karistirildigini gosterir.

### Oncelik (0-5)

- **Priority Exact**: Birebir eslesme. %62.7 - zor bir metrik, cunku komsu oncelikler birbirine yakin.
- **Priority +-1**: Bir komsuya toleransli. %94.0 - cogu hata kabul edilebilir sinirda.
- **Priority +-2**: Iki komsuya toleransli. %99.3 - neredeyse mukemmel.
- **Priority MAE**: Ortalama mutlak hata. 0.44 - ortalama yarim seviyeden az sapma.

### Guvenilirlik

- **ECE (Expected Calibration Error)**: Modelin verdigi confidence ile gercek dogruluk arasindaki fark. 0.068 cok iyi bir deger (0.0 mukemmel, >0.15 sorunlu). Dusuk ECE = "needs_review" esigi guvenilir calisir.
- **Avg Confidence**: Tum tahminlerin ortalama confidence degeri. 0.859.

## Orijinal Projeye Gore Farklar

| Orijinal | v1 |
|---|---|
| 3 suite (api, classifier, pipeline) + all | Sadece classifier |
| ONNX + PyTorch adapter | Sadece ONNX |
| 12 metrik fonksiyonu | 8 metrik (API'ye ozel 4 fonksiyon cikarildi) |
| 1843 satir Python | 1656 satir |
| API, pipeline section'lari raporda var | Sadece classifier section'i |

## Bagimliliklar

- `onnxruntime`: ONNX model inference
- `transformers`: HuggingFace tokenizer (sadece tokenizer, model degil)
- `numpy`: Softmax hesaplama ve metrikler
- `matplotlib`: HTML raporundaki grafikler (sadece report.py icin)

**Gerekmez:** `torch`, `sklearn`, harici CSS/JS dosyasi.

## Tasarim Kararlari

1. **Neden ABC?** `ClassifierAdapter` soyut sinifi sayesinde yeni bir model turu eklenince benchmark kodu degismez. Sadece `classify()` implemente edilir.

2. **Neden lazy import?** PyTorch kurulu olmayan sistemde `onnxruntime` calisir, ONNX kurulu olmayan sistemde `torch` calisir. Kutuphaneler adapter `__init__` edilene kadar yuklenmez.

3. **Neden kendi metriklerimiz?** sklearn hem agir bir bagimlilik hem de egitim amacli projede "hazir kutuphane kullanmak" yerine "metriklerin nasil calistigini gostermek" daha degerli.

4. **Neden self-contained HTML?** Harici CSS/JS/dosya yok. Tek bir `.html` dosyasi, email'le paylasilabilir, tarayiciyla acilir.

5. **Neden karanlik tema?** Modern dashboard standarti. Tailwind'in slate paletinden esinlenildi.

6. **Neden .onnx?** PyTorch modeli ONNX'e export edildi. ONNX Runtime CPU'da 2-3x daha hizli inference yapar, `torch` bagimliligi yoktur, tek bir dosya olarak tasinir.
