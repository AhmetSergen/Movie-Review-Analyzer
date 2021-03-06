# Movie-Review-Analyzer

This semantic analysis of IMDB movie commentary project on data mining subject was developed by:<br>
Ahmet Sergen Boncuk and Murat Mert Yurdakul.

## Açıklama [TR]

Veri madenciliği konusu üzerine IMDB film yorumlarının anlamsal analizinin yapıldığı bu proje:<br>
Ahmet Sergen Boncuk ve Murat Mert Yurdakul tarafından geliştirilmiştir.

### Giriş

Projenin amacı, IMDB filmlerine yapılan yorumların olumlu veya olumsuz olduğunu tahmin etmeye yarayan algoritmanın oluşturulmasıdır.
Yapılan yorumun olumlu veya olumsuz olmasının tespit edilip sayılması, bir filme gelen tepkilerin hangi yönde olduğunu anlamamıza yardımcı olabilir.

### Deneysel Kurulum

Projede kullanılan verinin işlenmemiş hali, kullanılacak algoritmalar için anlam çıkartılmayacak yapıdadır.<br>
Sınıf sayısı bu projede "olumlu" ve "olumsuz" olmak üzere 2 tanedir.<br>
Obje sayısı 12500 pozitif yorum ve 12500 negatif yorum olmak üzere toplam 25000 adettir. Her bir yorum bir obje olarak sayılır.<br>
Öznitelik sayısı tüm yorumlarda bulunan birbirinden farklı olan tüm kelimelerin toplam sayısı kadardır.<br>
Bu projede amaç, bu veri setini kullanarak en iyi tutarlılığa sahip analiz modelini oluşturmaktır.<br>

### Yapılan İşlemler

<ul>
  <li>Ön işleme (Preprocessing)</li>
  <ul>
      <li>HyperText etiketleri</li>
      <li>Büyük harfler</li>
      <li>Özel karakterler</li>
      <li>Yazım ve Heceleme Yanlışları (Typos-SpellCheck)</li>
      <li>Kökleştirme (Stemming)</li>
      <li>Etkisiz Kelimeler (Stopwords)</li>
  </ul>
  <li>Öznitelik Temsili (Feature Representation)</li>
  <ul>
    <li>Raw Term-Frequency</li>
    <li>N-Gram</li>
  </ul>
  <li>Öğrenme Algoritmaları (Learning Algorithms)</li>
  <ul>
    <li>Decision Tree (Karar Ağacı)</li>
    <li>Support Vector Machine (SVM)</li>
    <li>Naive Bayes (Basit Bayes)</li>
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

### Ön İşleme (Preprocessing)

Veri önişleme, model için kullanılacak verinin, ham veriden anlaşılabilir veriye dönüştürülme tekniğine denir.<br>
Öznitelik işlemlerinden önce kullanılacak olan verinin daha uygun, sade ve az karmaşık hale getirilmesi amaçlanır.<br>
İnternet ortamında yapılan, geneli amatör olan yorumlarda veri modeli için performans düşüklüğüne yol açacak unsurların olması kaçınılmazdır.<br>
Önişleme ile ham veri seti amacına daha uygun bir şekle getirilir.<br>

### HyperText Etiketleri

İnternet kaynaklarından alınan metinlerde, web programlamada kullanılan etiketler bulunabilir. <br>
Örneğin : &#60;h1&#62;Hello&#60;/h1&#62; <br>
Bu etiketlerin model oluşturmada bir değeri yoktur ve bu yüzden ayıklanması gereklidir.

### Özel Karakterler

Metinlerde kullanılan özel karakterler (noktalama işaretleri…) ve bu projede olmaması gereken Türkçe karakterler metin içerisinden çıkarılmıştır.

### Büyük Harfler

Modelin geliştirmesinde, bir harfin büyük veya küçük olmasının bir anlamı olmadığı için, büyük küçük harf farkının yeni niteliklerin oluşmasına yol açmasını önlemek için tüm karakterler küçük harfe çevrilir.

### Yazım ve Heceleme Yanlışları (Typos-Spellcheck)

Belirtilmek istenen ifadenin yazımı sırasında yapılan tuşlama hatalarıdır.<br>
Örneğin: "Awesome" yerine "Awsome" yazılması.<br>
Yazım hataları sebebiyle, işlenecek olan veri seti olması gerektiğinden fazla büyür ve gereksiz bir şekilde büyür.<br>
Örneğin: "Awesome" ve "Awsome" adında iki farklı nitelik ( feature ) oluşur. Fakat bunların iki ayrı nitelik olması bize bir kazanç sağlamaz.<br>
Bu durumu düzeltmeye yarayan hazır kütüphaneler bulunur.<br>
Örneğin: "Someting is hapening here" kelimesi pySpellChecker ile düzeltildiğinde "Something is happening here" şeklini alır.

### Kökleştirme (Stemming)

Yalın kelimeler cümle içerisinde kullanılırken gerekli ekler alabilir.<br>
Örneğin: "Enjoy" ve "Enjoyed" <br>
Fakat bu durum aynı anlama gelen kelimelerin farklı nitelikler ile değerlendirilmesine yol açar.<br>
Nitelik değerlendirmesi sırasında, bu projede önemli olan kelime kökleri olduğu için, tüm kelimelerin yalnızca köklerini ayıklanıp incelenir.<br>
Ortak köke sahip olan tüm kelimeler tek bir nitelik olarak değerlendirilir.<br>
Örneğin: "Love", "Loved", "Loves" = "Love"

### Etkisiz Kelimeler (StopWords)

Metinlerde anlatılmak istenen duruma vurgu yapan önemli kelimeler dışında, bolca kullanılan ama modelin etiketine hiçbir şekilde etkisi olmayan kelimeler bulunmaktadır.<br>
Örneğin: "The", "is", "a", "i", "am" … <br>
Bu kelimeler bize bir kazanç sağlamamakla beraber modelin performansını büyük oranda düşürecektir.<br>
Bu etkisiz kelimeler, modelin performansını iyileştirmek adına önişleme safhasında ayıklanır.<br>

### Öznitelik Temsili (Feature Representation)

Her veri seti, algoritmanın işleyebileceği şekilde verilmeyebilir.<br>
Sayısal bir değer yerine alfabe karakterlerinden oluşabilir.<br>
Algoritmanın veriyi anlamlandırabilmesi için veri modeline uygulanan işlemler öznitelik temsili olarak adlandırılır.<br>
IMDB veri seti 25000 farklı, algoritma için anlam taşımayan yorumdan oluştuğundan bu verileri sayısal bir değere çevirmek gereklidir.

### Raw Term-Frequency

TF = f(t,d)<br>
TF değeri, t teriminin d dökümanında kaç kez geçtiğidir.<br>
Bu algoritmada, önişlemden geçmiş niteliklerin her iki sınıfta da (pozitif negatif) kaç kez geçtiği bulunmuştur. Bu değerler bir tablo haline getirilmiştir.<br>
Sonrasında, kullanım sayıları iki sınıf arasında belirli bir oranda yakınlık gösteren nitelikler elenmiştir. Çünkü her iki sınıfta da yakın oranda bulunan nitelikler, birbiri ile aynı etkiye sahip olacağından, modelin geliştirilmesinde bir yardımı olmayacaktır.

### Olumsuzlaştırma Kelimeleri (Negative Words) ile özelleştirilmiş N-Gram

N-Gram, n sayıda kelimenin birleştirilerek tek bir nitelik olarak sayılmasıdır.<br>
Bir kelimenin olumlu ile olumsuz hali tamamen farklı nitelikler olarak sayılmalıdır. İngilizcede olumsuzlaştırma durumu genelde iki kelime ile belirtilir.<br>
Örneğin : "Good" ile "Not good" tamamen farklı nitelikler olarak değerlendirilir.<br>
Olumsuzlaştırma kelimeleri ile bundan sonra gelen kelime birleştirilerek tek bir nitelik olarak sayılmıştır.

### Öğrenme Algoritmaları

İşlenen verinin sonucunda oluşmuş olan modele anlam yüklemeye yarayan yöntemlerdir.<br>
Kullanılan öğrenme algoritmaları:
<ul>
  <li>Decision Tree (Karar</li>
  <li>Support Vector Machine (SVM)</li>
  <li>Naive Bayes (Basit Bayes</li>
  <li>K-Nearest Neighbor</li>
  <li>Random Forest</li>
  <li>K-Means</li>
</ul>

### Performans Değerlendirmesi

12500 olumlu ve 12500 olumsuz yorum ile eğitilen algoritma üzerinde uygulanan bir örnekte Doğruluk (Accuracy) değerleri şu şekildedir:
<ul>
<li>Decision Tree : 0.72</li>
<li>SVM : 0.93</li>
<li>Naive Bayes : 0.98</li>
<li>KNN : 0.61</li>
<li>Random Forest : 0.61</li>
<li>K-Means : 0.18</li>
</ul>
