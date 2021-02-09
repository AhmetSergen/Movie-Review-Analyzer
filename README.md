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

####Ön işleme
<ul>
  <li>HyperText etiketleri</li>
</ul>
