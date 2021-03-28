#Explain your work


"""

Öncelikle verdiğiniz eğitim için çok teşekkür ederim. Gerçekten makine öğrenmesi hakkında başlangıç noktası oluşturmak için çok verimli bir eğitimdi benim için. 

Cevaplarımı zamanın yetersiz olamsından dolayı türkçe yazmak zorunda kaldım. Umarım anlayış gösterirsiniz. Talep etmeniz durumunda cevapları ingilizce olarak ta hazırlayabilirim. 

Ayrıca Phyton dilinde şimdiye kadar hiç proje geliştirmedim. O yüzden yazdığım kodlar biraz makarna görünürse kusura bakmayın. Olabildiğince yorum satırları ile neyi neden yaptığımı açıklamya çalıştım.

Eğer ukalalık olarak görmezseniz, bir iki nacizane tavsiyem olacak.

Her ne kadar yazılım öğrenmenin temeli ingilizce bilmek olsada, olabildiğince türkçe eğitimler hazırlamanız iyi olacaktır. Ana dili ingilizce olan birinin en azından başlangıç seviyesi bir eğitimde, T0 anından 1-0 öne geçmesi bence Türkiyede yazılım sektörünün 2000 lerde kalmasının en önemli sebebidir.

Bir de homework ve final project için verilen süreleri en azından benim gibi özel sektörde çalışan insanlarıda gözeterek tanımlarsanız sevinirim. Öğrencilik döneminde kişinin bu tarz eğitimlere ayıracak çok zamanı olabiliyor fakat çalışan kesim her zaman gerekli zamanı arttıramıyor. Ben bu ödevlere pazar günü 6 da başlayabildim ve eminim ki benim gibi çalışan kesimden bir çok kişi de bu ödevlere anca zaman bulabilmiştir. Bu yüzden bu süreci biraz daha uzatarak eğitime katılan öğrencilerin, daha iyi çıktılar öğretmesini ve verdiğiniz eğitimden daha iyi faydalanmalarını sağlayabilirsiniz.


"""

#Question 1


#1) How would you define Machine Learning?  
"""
 A1) Makine öğrenmesi, makineye verilen verieri, verilerin türüne ve istenilen sonuçlara göre belirlenecek algoritmalar kullanarak, verilerin nasıl anlamlandırılacağını ve nasıl yorumlayacağını öğretme ve öğrendiklerini yeni vereler ile test etme ve geliştirme 
 sürecedir.



""" 
#2) What are the differences between Supervised and Unsupervised Learning? Specify example 3 algorithms   for each of these. 

"""

A2)  Supervised Learning, çıktıları bilinen (Labeled) etiketlenmiş veriler ile makineyi eğitmektir. Bu öğrenme yöntemi daha çok daha önceden öğretilmiş verilerle oluşturulan modeller kullanarak, yeni verilerin sonuçlarını tahmin etmede veya karar almada kullanılır. 

Unsupervised Learning ise çıktıları bilinmeyen(Unlabeled) etiketlenmemiş veri ile makineyi eğitmektir. Bu öğrenme yintemi ise daha çok nasıl sınıflandırılacağı öngörülemeyen büyük veri setlerinde, en doğru sınıflandırılmanın belirlenmesi ve yeni gelicek verilerinde bu sınıflandırma modelleri kullanılaralk sınıflandırılmasının sağlanmasıdır.
 
"""

#3) What are the test and validation set, and why would you want to use them? 

"""

A3) Makine öğrenmesi sürecinde, makineye öğretilmesi planlanan verinin, eni iyi modeli geliştirebilmek amacıyla, train, validation ve test olarak bölümlenmesi gerekir.

Bu bölümlemede "train" eğitim verisi, "validation" eğitilen modelin doğrulanmasında kullanılacak veri ve test veirsi is modelin test edilerek doğruluğunun ölçüleceği veri olarak açıklanabilir.

Validation ve Test verisi arasındaki en belirgin fark, validation verisinin öğrenme süreci içinde kullanılması, test verisinin ise öğrenme süreci tamamlandıktan ve model geliştirildikten sonra kullanılmasıdır.

Validation ve Test Verilerinin kullanılması, en iyi modeli geliştirebilmek için modelin çalışma performansının ölçülebilmesi ve sonuçlara göre aksiyonlar alrak daha iyi bir model hazırlanması için gereklidir.


"""
#4) What are the main preprocessing steps? Explain them in detail. Why we need to prepare our data? 

"""

A4) Veri hazırlama süreci, elimizdeki verinin, analiz edilmesi, temizlenmesi, eksik verilerin tamamlanması, anomalilerin düzeltilmesi, ve verinin makinenin anlayabileceği şekilde ölçeklendirilmesi, gruplanması, makina verisine çevrilmesi işidir. En iyi modeli geliştirmek için eğitimde kullanılacak olan veri setinin iyi hazırlanmış olması gerekir.

Veri Hazırlama Süreçleri.

1 - Tekrar eden verilerin temzilenmesi.

Eğitimde kullanılacak olan verilerin içinde bir verinin birden falza yer almasını önlemek ve bu veririnin tüm veri seti üzerindeki olasılığının yanlış hesaplanmasının önüne geçmek amacıyla tekrar edilen verilerin silinmesi işlemidir.

2 - Eşit Dağılmamış Veriler

Elimizdeki verilerde bir çıktının örneklerinin diğer çıktıların örneklerine kıyasla daha fazla veya az olması durumudur. Bu durum geliştirilecek modelde az örneği az olan çıktının olasılığını düşürecek ve yanlış bir model eğitilmesine sebep olabilcektir.

Bu durumda yapılacak işlem, örneği az olan çıktının örneğini arttırarak veya örneği çok olan çıktının örneğini azaltarak, çıktılar arasında denge sağlanmalıdır.

3 - Eksik Kayıp Veriler

Eğitimde kullanılacak danaın içinde boş veya hatalı veriler olabilir. Bu veriler eğitime başlanmadan önce doldurulmalı veya temizlenmelidir. Doldurulmak istenen eksik verinin diğer veriler kullanılarak histogramı çizilmeli ve histogram göre mean veya median değeri kullanılarak bu veirler doldurulmalıdır.

4 - Anomali Bulma

Eğitimde kullanılacak verinin içinde anormal düşük veya anormal yüksek verilerin bulunması ve bu verilerin temizlenmesi işlemidir. Bu verilerin temizlenmemesi, veri setinin ortalamsını bozarak modelin yanlış verileri yanlış anlamlandırmasına yol açabilir. Bu verilerin belirlenmesinde, Standar Sapma - IQR Hesaplaması veya Isolation Forest yöntemleri kullanılabilir.

5 - Girdi Ölçeklendirme

Eğitimde kullanılacak verilerin ölçeklendirilmesi ve bu şekilde veriler arsanındaki farkların standardize edilmesi işlemidir. 
Bu işlemde standardizasyon veya normalizasyon teknikleri kullanılabilir. 


6 - Veri Gruplandırma

Eğitimde kullanılacak verilerin gruplandırılıması ve bu grup içindeki verilerin tamamına hesaplanan verinin atanması işlemdir. Bu işlem verinin anlamlandırılmasında bu verileri arsındaki farklardan doğacak hataların önlenmesi için kullanılabilir. 



7 - Girdi Azaltma

Eğitimde kullanılacak verilerin arasında ilşikiler kurarak bu veriler ile yeni veri oluşturma ve kullanılan verilerin yerine yeni verinin eklenmesiyle girdilerin azaltılmasını sağlamak amacıyla yapılan işlemdir.


8 - Feature Encoding 

Eğitimde kullanılacak verinin makinenin öğrenebileceği dile çevrilemsi işlemidir. Dönüştürlecek verinin sıralı bir değer olması veya olammasına göre One-Hot Encoding veya sıralı numaralandırma işlmeleri yapılabilir.


9 - Veri bölümleme

Eğitimde kullanılacak verinin, geliştirilen modelin doğrulanması ve test edilemsi amacıyla eğitim, doğrulama ve test  veya eğitim ve test olarak bölümlenmesi işlemdir. Bu bölümlemeyi neden yaptığımızı yukarıda detaylı olarak anlatmıştık (Bkz: A3 :) )


"""
#5) How you can explore countionus and discrete variables? 

"""

A5) Devamlı veya kesikli verilerin incelenmesinde kullanılması gereken yöntem grafik çizilmesidir. Devamlı veriler için histogramlar, kesikli veriler içinde sütun grafikler tercih edilebilir. 

"""
#6) Analyse the plot given below. (What is the plot and variable type, check the distribution and make comment about how you can preproccess it.) 

"""

A6) Verilen grafik devamlı bir veri için çizilmiş bir histogramdır. Sadece bu veriyi kullanrak bir model geliştirmem istenseydi, bu veri üzeinde yapacağım işlemler, tekrar eden verilerin temizlenmesi ve anomalilerin bulunması işlemi olurdu.

"""
