# KRunner Semantic Search (EmbeddingGemma 2)

KDE Plasma KRunner için **Google DeepMind EmbeddingGemma 2** çok modlu (multimodal) gömme modelini kullanan yerel, gizlilik odaklı ve hızlı semantik dosya arama eklentisi ve arka plan dizinleme servisi.

Klasik anahtar kelime eşleştirmesi yapan arama araçlarının (Baloo vb.) aksine; aradığınız dosyanın adını tam bilmeseniz bile içeriğini, anlamını veya görsel/işitsel tanımını yazarak dosyalarınıza anında erişmenizi sağlar.

---

## 🎯 Proje Hedefleri ve Özellikler

1. **Doğal Dil ile Arama:** Dosya adı hatırlanmasa dahi anlam bazlı arama (Örn: `"evcil hayvan maddesi"`, `"mavi gömlekli profil fotoğrafı"`).
2. **KDE Plasma 6 / 5 Uyumlu KRunner D-Bus Arayüzü:** KRunner'ı yavaşlatmayan, harici D-Bus servisi (`org.kde.krunner1`).
3. **%100 Yerel ve Gizli:** Tüm indeksleme ve sorgulama bilgisayarınızda yerel çalışır, internete hiçbir veri çıkışı yapılmaz.
4. **Hafif ve Hızlı Çıkarım:** `llama.cpp` (veya `llama-cpp-python`) ile CPU/GPU üzerinden ~50ms yanıt süresi.
5. **Gömülü Vektör Veritabanı:** Sunucu gerektirmeyen gömülü `sqlite-vec` (veya `sqlite` cosine similarity) depolaması.

---

## 🏗️ Mimari Tasarım

Sistem iki ana bileşenden oluşur:

```
[ Kullanıcı (KRunner) ]
          │  (D-Bus: org.kde.krunner1)
          ▼
┌────────────────────────────────────────────────────────┐
│  krunner-semantic-daemon (D-Bus Servisi)               │
│  - Match(query): Query'yi vektörleştirir              │
│  - EmbeddingGemma 2 Text Encoder (llama.cpp)           │
│  - SQLite-Vec üzerinden en yakın komşuları sorgular     │
│  - KRunner'a sonuçları, skorları ve ikonları iletir     │
└────────────────────────────────────────────────────────┘
          ▲
          │ (Okur)
┌─────────┴──────────────────────────────────────────────┐
│  ~/.local/share/krunner-semantic/index.db              │
└────────────────────────────────────────────────────────┘
          ▲
          │ (Yazar)
┌─────────┴──────────────────────────────────────────────┐
│  krunner-semantic-indexer (Arka Plan Dizinleyici)     │
│  - Belirlenen klasörleri izler (~/Documents vb.)       │
│  - Dosyaları parçalar (Metin, PDF, Görsel)             │
│  - EmbeddingGemma 2 ile 768-dim vektör çıkarır         │
│  - Veritabanına kaydeder/günceller                     │
└────────────────────────────────────────────────────────┘
```

---

## 📁 Proje Dizin Yapısı

```
krunner-semantic-search/
├── README.md                          # Bu doküman
├── ROADMAP.md                         # Aşama aşama geliştirme planı
├── setup.py / pyproject.toml          # Bağımlılıklar ve paketleme
├── desktop/
│   └── plasma-runner-semantic.desktop # KRunner plugin kayıt dosyası
├── systemd/
│   └── krunner-semantic.service       # Kullanıcı systemd servis dosyası
└── src/
    ├── __init__.py
    ├── daemon.py                      # org.kde.krunner1 D-Bus sunucusu
    ├── indexer.py                     # Dosya tarama ve indeksleme motoru
    ├── embedder.py                    # EmbeddingGemma 2 llama.cpp entegrasyonu
    ├── storage.py                     # SQLite-Vec veri yönetim katmanı
    └── extractors/                    # Dosya ayrıştırıcıları
        ├── text.py                    # Düz metin / Markdown / Kod
        ├── pdf.py                     # PDF metin çıkarımı
        └── image.py                   # Görsel işleme / yeniden boyutlandırma
```

---

## 🚀 Başlangıç & Gereksinimler

- Python 3.10+
- KDE Plasma 5 veya 6
- Gerekli Sistem Kütüphaneleri: `python-dasbus` (veya `dbus-next`), `llama-cpp-python`, `sqlite-vec`
