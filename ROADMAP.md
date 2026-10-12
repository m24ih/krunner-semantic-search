# 🗺️ KRunner Semantic Search Geliştirme Yol Haritası (ROADMAP)

Bu belge, projeyi sıfırdan bitmiş bir KRunner eklentisine dönüştürmek için uygulanacak adım adım teknik aşamaları içerir.

---

## 📌 Faz 1: D-Bus Altyapısı ve KRunner İletişimi (Skeleton)
- [ ] `desktop/plasma-runner-semantic.desktop` şablonunu oluştur.
- [ ] `src/daemon.py` içinde `org.kde.krunner1` arayüzünü ayağa kaldır (`Actions`, `Match`, `Run` metodları).
- [ ] KRunner ile sahte (mock) sonuçlar dönerek entegrasyonu test et (`qdbus` ve KRunner UI üzerinden doğrula).
- [ ] `systemd/user/krunner-semantic.service` oluşturarak arka planda servis olarak çalışabilmesini sağla.

## 📌 Faz 2: Vektör Depolama ve Veritabanı (`src/storage.py`)
- [ ] SQLite veritabanı şemasını tasarla:
  - `files` tablosu: `id`, `filepath`, `mtime`, `hash`, `kind`
  - `chunks` tablosu: `id`, `file_id`, `chunk_index`, `content_snippet`, `embedding` (768-dim float)
- [ ] `sqlite-vec` entegrasyonunu yap (vektörler arası Cosine Distance sorgusu için KNN arama).
- [ ] Hızlı sorgu için limit ve threshold (alaka eşiği) filtreleme mekanizması kur.

## 📌 Faz 3: EmbeddingGemma 2 Çıkarım Motoru (`src/embedder.py`)
- [ ] Model GGUF dosyasının indirilmesi / konumu (`~/.local/share/krunner-semantic/models/`).
- [ ] `llama-cpp-python` veya bağımsız `llama.cpp` embedding runner'ı ile Python sarmalayıcısı yaz.
- [ ] Metin sorgularını 768 boyutlu vektöre dönüştürme fonksiyonu (`embed_query(text)`).
- [ ] Çoklu metinleri toplu vektörleştirme (`embed_batch(texts)`).
- [ ] Performans ve bellek optimizasyonu (yalnızca text encoder'ı bellekte tutma).

## 📌 Faz 4: Çıkarıcılar ve Dizinleme Motoru (`src/indexer.py`)
- [ ] `src/extractors/text.py`: Kod, Markdown, `.txt` dosyalarını belirli token/karakter pencerelerine (chunking) bölme.
- [ ] `src/extractors/pdf.py`: PDF sayfalarından metin çekme (`pypdfium2` veya `pymupdf`).
- [ ] `src/extractors/image.py`: Görsellerin modele beslenmesi için boyutlandırma.
- [ ] CLI ile ilk dizinleme komutunu oluştur: `python -m src.indexer --folder ~/Documents`.
- [ ] Dosya değişikliklerini (mtime/hash) izleyerek sadece değişen dosyaları yeniden indeksleyen akıllı güncelleme mekanizması.

## 📌 Faz 5: Uçtan Uca Entegrasyon ve Test
- [ ] `daemon.py`'ı `embedder.py` ve `storage.py`'a bağla.
- [ ] KRunner'a `sem: arama kelimesi` yazıldığında milisaniyeler içinde doğru dosyaların ve sayfaların listelendiğini doğrula.
- [ ] `Run` fonksiyonu ile dosyaya tıklandığında `xdg-open` ile ilgili sayfa/konumda açılmasını sağla.

## 📌 Faz 6: Paketleme & KDE Store (Opendesktop) Dağıtımı
- [ ] `install.sh` kurulum scripti yaz (KRunner `.desktop` dosyasını `~/.local/share/krunner/dbusplugins/` dizinine kopyalar).
- [ ] KDE Store (`store.kde.org`) KRunner kategorisine uygun `.tar.gz` paketi ve ekran görüntüleri hazırla.
