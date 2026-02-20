# Softmax Tabanlı İstemci Taraflı Yük Dengeleyici

## 📌 Proje Özeti

Bu projede, K adet zamanla performansı değişen (non-stationary) ve gürültülü (noisy) sunucudan oluşan bir dağıtık sistem için istemci taraflı bir yük dengeleme algoritması geliştirilmiştir.

Amaç, toplam bekleme süresini (latency) minimize etmek, yani toplam ödülü (reward) maksimize etmektir.

Klasik Round Robin ve Random algoritmaları yerine, geçmiş performans verisini kullanarak olasılıksal seçim yapan **Softmax Action Selection** algoritması uygulanmıştır.

---

## 🧠 Problem Tanımı

Her sunucu:

- Zamanla değişen ortalama gecikmeye (drift) sahiptir
- Gaussian gürültü içerir
- Gerçek dağıtık sistem belirsizliğini simüle eder

Bu nedenle problem, **Non-Stationary Multi-Armed Bandit** problemi olarak modellenebilir.

Statik algoritmalar bu ortamda adaptasyon gösteremez.

---

## ⚙️ Gerçekleştirilen Algoritmalar

### 1️⃣ Round Robin
- Sunucuları sırayla seçer
- Öğrenme yapmaz
- Adaptif değildir

### 2️⃣ Random
- Rastgele seçim yapar
- Geçmiş performansı kullanmaz
- Adaptif değildir

### 3️⃣ Softmax Action Selection
- Her sunucu için bir Q değeri tutar
- Q değeri geçmiş ödüllerin ortalamasıdır
- Olasılıksal seçim yapar

Seçim olasılığı:

P(i) = exp(Q_i / T) / Σ exp(Q_j / T)

Burada:
- Q_i → i. sunucunun tahmini ödülü
- T → temperature parametresi (exploration–exploitation dengesi)

---

## 🔥 Neden Softmax?

Softmax algoritması:

- Adaptif öğrenme yapar
- Exploration–exploitation dengesini sağlar
- Non-stationary ortamlarda daha iyi performans gösterir
- Geçmiş veriye dayalı olasılıksal karar verir

Round Robin ve Random algoritmaları ise öğrenme yapmadığı için dinamik ortamlarda verimsizdir.

---

## 🧮 Nümerik Stabilite Problemi

Softmax hesaplamasında doğrudan:

exp(Q)

kullanımı büyük Q değerlerinde overflow hatasına yol açabilir.

Bu problemi önlemek için:

exp(Q - max(Q))

yöntemi uygulanmıştır.

Bu teknik literatürde **Log-Sum-Exp Trick** olarak bilinmektedir ve sayısal taşmayı engeller.

---

## ⏱ Çalışma Zamanı Analizi

Her seçim adımında:

- Maksimum Q değeri bulma → O(K)
- Üstel hesaplama → O(K)
- Normalize etme → O(K)

Dolayısıyla her adım:

O(K)

Toplam simülasyon karmaşıklığı:

O(T × K)

Burada:
- T → zaman adımı sayısı
- K → sunucu sayısı

---

## 📊 Sonuçlar

Simülasyon sonuçlarına göre:

- Softmax algoritması zamanla daha iyi performans gösteren sunuculara daha yüksek olasılık atamaktadır.
- Toplam reward açısından Round Robin ve Random algoritmalarından daha iyi sonuç vermektedir.
- Dinamik ortamlarda adaptif algoritmaların üstünlüğü gözlemlenmiştir.

Grafik çıktısı cumulative reward üzerinden karşılaştırma sunmaktadır.

---

## 🚀 Çalıştırma Talimatları

Gerekli kütüphaneler:

```bash
pip install numpy matplotlib
