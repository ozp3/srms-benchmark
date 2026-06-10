"""
Metrikler: accuracy, precision, recall, F1, confusion matrix, calibration.
sklearn bagimliligi yok — saf Python + standard lib.
v1 — yalnizca classifier metrikleri.

Bu dosya, modelin tahminlerini sayisal olarak degerlendirmek icin kullanilan
tum metrik fonksiyonlarini icerir. Her fonksiyon bagimsiz calisir, disariya
bagimliligi yoktur. Test sirasinda suites.py bu fonksiyonlari cagirir.
"""


# =============================================================================
# TEMEL SINIFLANDIRMA METRIKLERI
# =============================================================================

def accuracy(y_true: list, y_pred: list) -> float:
    """
    Dogruluk (accuracy): dogru tahmin edilenlerin toplam ornek sayisina orani.
    En temel basari olcusu. 0.0 (hic dogru yok) ile 1.0 (hepsi dogru) arasinda.

    Formul: dogru_tahmin_sayisi / toplam_ornek_sayisi

    Neden kullaniyoruz: Ilk bakista modelin genel performansini gormek icin.
    Tek basina yeterli degildir — sinif dengesizliginde yaniltici olabilir.
    """
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(y_true: list, y_pred: list, labels: list | None = None) -> dict:
    """
    Karmaşıklık matrisi (confusion matrix): her gercek sinifin hangi tahmin
    sinifina dustugunu sayan 2 boyutlu matris.

    Return: {gercek_sinif: {tahmin_sinif: sayi}}
    Ornek: {"road_damage": {"road_damage": 15, "waste": 2}} → 15 dogru, 2 yanlis waste'e kaymis.

    Neden kullaniyoruz: Modelin hangi siniflari birbirine karistirdigini gormek icin.
    Ozellikle benzer kategoriler (orn. road_damage ↔ sidewalk_damage) arasindaki
    karisikligi tespit etmek icin kritik.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    return cm


def per_class_prf(y_true: list, y_pred: list, labels: list | None = None) -> dict:
    """
    Sinif bazinda Precision, Recall, F1 ve Support (ornek sayisi) hesaplar.

    Precision: "X sinifi dediklerimin kaci gercekten X?" (yanlis pozitif azsa yuksek)
    Recall:    "Gercek X'lerin kacini yakaladim?"      (yanlis negatif azsa yuksek)
    F1:        Precision ve Recall'un harmonik ortalamasi (dengeli tek olcu)
    Support:   O sinifa ait gercek ornek sayisi

    Neden kullaniyoruz: Her sinifin ayri ayri performansini gormek icin.
    Ozellikle "irrelevant" ve "normal" gibi zor siniflarda modelin ne kadar
    basarili oldugunu anlamak icin onemli.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    out = {}
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        support = sum(1 for t in y_true if t == lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
    return out


def macro_f1(per_class: dict) -> float:
    """
    Macro F1: tum siniflarin F1 skorlarinin duz (aritmetik) ortalamasi.
    Her sinifa esit agirlik verir — az ornekli siniflar da cok ornekli siniflar
    kadar etkilidir.

    Neden kullaniyoruz: Sinif dengesizligi oldugunda (ornegin "irrelevant" 55
    ornek, "sewage_water" 17 ornek), modelin kucuk siniflari da ogrenip
    ogrenmedigini anlamak icin. Weighted F1 ile birlikte degerlendirilir.
    """
    if not per_class:
        return 0.0
    return sum(v["f1"] for v in per_class.values()) / len(per_class)


def weighted_f1(per_class: dict) -> float:
    """
    Weighted F1: her sinifin F1 skorunu o sinifin ornek sayisiyla (support)
    agirliklandirarak hesaplanan ortalama. Cok ornekli siniflar daha etkilidir.

    Neden kullaniyoruz: Gercek hayatta bazi siniflar daha sik gorulur. Bu metrik
    "toplamda kac ornekte basariliyiz?" sorusuna daha iyi cevap verir.
    Macro F1 ile arasindaki fark buyukse sinif dengesizligi var demektir.
    """
    total = sum(v["support"] for v in per_class.values())
    if total == 0:
        return 0.0
    return sum(v["f1"] * v["support"] for v in per_class.values()) / total


# =============================================================================
# ONCELIK (PRIORITY) METRIKLERI
# =============================================================================

def priority_accuracy(y_true: list[int], y_pred: list[int], tolerance: int = 0) -> float:
    """
    Oncelik tahmini dogrulugu. tolerance=0 birebir eslesme ister,
    tolerance=1 ±1 sapmaya izin verir, tolerance=2 ±2 sapmaya izin verir.

    Ornek: Gercek oncelik 4, tahmin 3 ise:
      - tolerance=0: yanlis  (4 != 3)
      - tolerance=1: dogru   (|4-3| <= 1)
      - tolerance=2: dogru   (|4-3| <= 2)

    Neden kullaniyoruz: Oncelik 0-5 arasi bir deger. 4 yerine 3 demek (High
    yerine Moderate) ciddi bir hata degil, rapor yine de islem gorur. Ama 4
    yerine 0 demek (High yerine Irrelevant) ciddi hatadir. Toleransli accuracy
    bu "kabul edilebilir sapma"yi olcer.
    """
    if not y_true:
        return 0.0
    return sum(abs(t - p) <= tolerance for t, p in zip(y_true, y_pred)) / len(y_true)


def priority_mae(y_true: list[int], y_pred: list[int]) -> float:
    """
    Mean Absolute Error (ortalama mutlak hata): oncelik tahminindeki ortalama
    sapma miktari. 0.0 = mukemmel, yuksek deger = buyuk sapma.

    Ornek: 5 ornek icin [4,4,4,4,4] tahmini gercek [4,3,5,4,4] ise
    MAE = (|4-4| + |4-3| + |4-5| + |4-4| + |4-4|) / 5 = (0+1+1+0+0)/5 = 0.4

    Neden kullaniyoruz: Priority accuracy sadece "tolerans icinde mi" der,
    MAE ise "ortalama kac birim sapma var" der. Daha hassas bir olcudur.
    """
    if not y_true:
        return 0.0
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


# =============================================================================
# GUVENILIRLIK / KALIBRASYON
# =============================================================================

def expected_calibration_error(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE): modelin verdigi confidence (guven) degeri
    ile gercek dogruluk arasindaki farki olcer. 0.0 mukemmel kalibrasyon,
    yuksek deger kotu kalibrasyon demektir.

    Nasil calisir:
    1. Tahminleri confidence'a gore 10 esit araliga (bin) bol
    2. Her bin icin: ortalama confidence - gercek accuracy = kalibrasyon hatasi
    3. Bin'leri ornek sayisina gore agirliklandirarak topla

    Ornek: Model 100 ornek icin %90 confidence verdi ama sadece %70'i dogru
    ciktiysa → ECE = 0.20 (asiri ozguvenli, kalibrasyon bozuk).

    Neden kullaniyoruz: "needs_review" esigi (0.60) icin kritik. Model "bundan
    eminim" dediginde gercekten emin olmali. ECE dusukse confidence degerine
    guvenebiliriz, "needs_review" karari saglikli calisir.
    """
    if not confidences:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, correct in zip(confidences, correctness):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, correct))

    total = len(confidences)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc  = sum(1 for _, ok in b if ok) / len(b)
        weight   = len(b) / total
        ece += weight * abs(avg_conf - avg_acc)
    return ece
