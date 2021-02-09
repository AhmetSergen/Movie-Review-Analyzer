# Movie-Review-Analyzer
Semantic Analysis For IMDB Movie Reviews

## Açıklama [TR]

Veri madenciliği konusu üzerine IMDB film yorumlarının anlamsal analizinin yapıldığı bu proje:
Ahmet Sergen Boncuk ve Murat Mert Yurdakul tarafından geliştirilmiştir.

### Giriş

Projenin amacı, IMDB filmlerine yapılan yorumların olumlu veya olumsuz olduğunu tahmin etmeye yarayan algoritmanın oluşturulmasıdır.
Yapılan yorumun olumlu veya olumsuz olmasının tespit edilip sayılması, bir filme gelen tepkilerin hangi yönde olduğunu anlamamıza yardımcı olabilir.

### Deneysel Kurulum

Projede kullanılan verinin işlenmemiş hali, kullanılacak algoritmalar için anlam çıkartılmayacak yapıdadır. 
Sınıf sayısı bu projede "olumlu" ve "olumsuz" olmak üzere 2 tanedir.
Obje sayısı 12500 pozitif yorum ve 12500 negatif yorum olmak üzere toplam 25000 adettir. Her bir yorum bir obje olarak sayılır.
Öznitelik sayısı tüm yorumlarda bulunan birbirinden farklı olan tüm kelimelerin toplam sayısı kadardır.
Bu projede amaç, bu veri setini kullanarak en iyi tutarlılığa sahip analiz modelini oluşturmaktır.

### Yapılan İşlemler

<ul>
  <li>Ön işleme (Preprocessing)</li>
  <ul>
      <li>HyperText etiketleri</li>
      <li>Büyük harfler</li>
      <li>Özel karakterler</li>
      <li>Yazım ve Heceleme Yanlışları Typos Spell Check</li>
      <li>Kökleştirme ( Stemming</li>
      <li>Etkisiz Kelimeler ( Stopwords</li>
  </ul>
  <li>Öznitelik Temsili (Feature Representation)</li>
  <ul>
    <li>Raw Term-Frequency</li>
    <li>N-Gram</li>
  </ul>
  <li>Öğrenme Algoritmaları (Learning Algorithms)</li>
  <ul>
    <li>Decision Tree (Karar</li>
    <li>Support Vector Machine (SVM)</li>
    <li>Naive Bayes (Basit Bayes</li>
    <li>K-Nearest Neighbor</li>
    <li>Random Forest</li>
    <li>K-Means</li>
  </ul>
  <li>Performans Değerlendirmesi (Performance Parameters)</li>
  <ul>
    <li>Confusion Matrix</li>
    <li>Accuracy</li>
    <li>Precision</li>
    <li>Recall</li>
    <li>F-Measure</li>
  </ul>
</ul>
