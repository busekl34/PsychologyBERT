import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder="templates")

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")

sentiment_model_path = r"C:\Users\busek\BertPsikolojikAsistan\psikolojik_asistan_ince_ayar_buse_kilic"

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    sentiment_model_path,
    local_files_only=True
)

etiketler = [
    "İntihar Düşünceleri",
    "Yeme Bozuklukları",
    "Uyku Bozuklukları",
    "Cinsel Bozukluklar",
    "Bağımlılıklar",
    "Öfke Kontrol Bozuklukları",
    "Borderline",
    "Psikosomatik Bozukluklar",
    "Obsesif Kompulsif Bozukluk",
    "Çocuklarda Davranış Bozuklukları",
    "Depresyon ve İlişkili Bozukluklar",
    "Aile ve İlişki Sorunları",
    "Dikkat Eksikliği ve Hiperaktivite Bozukluğu",
    "Travma"
]

label_encoder = LabelEncoder()
label_encoder.fit(etiketler)

destek_mesajlari = {
    "İntihar Düşünceleri": "🧠 Hayat bazen ağırlığını fark ettirmeden çöker omuzlara, insan nefes almakta bile zorlanır. Ama bil ki bu karanlıkta bile bir umut ışığı vardır, ve sen o ışığı bulabilecek güce sahipsin. Hissettiğin duygular geçici olabilir; acının içinden geçen yollar zamanla iyileşir, ve bir gün bu günlere dönüp baktığında, nasıl güçlendiğini görebileceksin. Seni önemseyen, duygularını anlamaya çalışan insanlar var; yalnız değilsin. Bir adım atmak her şeyi değiştirebilir—ister bir kelime, ister bir sessizlik olsun, bu dünyada yerin var ve değerlisin.",
    
    "Yeme Bozuklukları": "🌸 Yeme bozuklukları, kişinin bedenine ve ruhuna zarar veren sessiz bir savaş olabilir. Bu savaşta yalnız kalmak zorunda değilsin. Hissettiklerin gerçek ve önemli—kendini suçlamak ya da utandırmak yerine, bu sürecin profesyonel bir destekle iyileştirilebileceğini bilmek çok önemli. Senin için uygun olan terapi yöntemleriyle ve uzman desteğiyle, bu karanlık döngü kırılabilir. İyileşmek zaman alabilir ama her adım seni kendine biraz daha yaklaştırır. Yardım istemek güçsüzlük değil, içindeki gücün bir göstergesidir.",
    
    "Uyku Bozuklukları": "🌙 Uykusuzluk, gün içinde yaşanan zihinsel ve fiziksel tükenmişliğe yol açabilir—ama bil ki bu döngü kırılabilir. Uykuyla ilgili sorunlar çoğu zaman stres, kaygı veya hormonal dengesizliklerden kaynaklanabilir. Bu durum seni yalnız hissettirebilir, ama çözüm yalnız başına aranmak zorunda değil. Bir uzmana danışarak, sana özel önerilerle bu sorunun köküne inmek mümkündür. Düzenli rutinler, gevşeme teknikleri ve gerekirse medikal destek, uyku kaliteni iyileştirmende yardımcı olabilir.",
    
    "Cinsel Bozukluklar": "❤ Cinsel sağlık, hem bedensel hem de duygusal iyi oluşun temel parçalarından biridir. Kendi bedenini tanımak, sınırlarını bilmek ve bu konuda açık iletişim kurabilmek sağlıklı ilişkiler kurmanın anahtarıdır. Güvende hissetmek, kendine ve partnerine saygı duymak, korunma yöntemlerini öğrenmek ve gerektiğinde profesyonel destek almak—hepsi cinsel sağlığını destekler. Unutma, bu konuya özen göstermek hem kendine duyduğun saygının hem de yaşam kalitenin bir göstergesidir.",
    
    "Bağımlılıklar": "🌀 Bağımlılık, insanın bedenine ve ruhuna zincirler vurabilir—ama bu zincirler kırılabilir. Her neyle ilgili olursa olsun (madde, teknoloji, ilişki…), bağımlılık genellikle içsel bir boşluğu doldurma çabasıdır. Bu durumla tek başına baş etmek zorunda değilsin. Profesyonel destek almak, hem nedenleri anlamak hem de sağlıklı başa çıkma yollarını öğrenmek için çok kıymetlidir. Psikolojik destek, grup terapileri, uzman klinikler bu sürecin en güçlü yol arkadaşları olabilir.",
    
    "Öfke Kontrol Bozuklukları": "🔥 Öfke kontrolü zorlayıcı olabilir; kişi içindeki patlamaları bastırmaya çalışırken daha da yorulabilir. Bu duygunun altında çoğu zaman anlaşılmama, haksızlık ya da geçmiş deneyimlerin yarattığı içsel çatışmalar yatabilir. Ama öfke kötü bir şey değildir; onu nasıl yönettiğin asıl farkı yaratır.Bu süreçte bir uzmandan destek almak, öfkenin kökenlerini anlamana ve kendini daha sağlıklı ifade etme yolları bulmana yardımcı olabilir. Nefes egzersizleri, zamanlama stratejileri ve terapötik teknikler öfkeni bastırmadan yönetmeni sağlar. Yardım istemek zayıflık değil; içsel huzuru aradığının güçlü bir göstergesidir.",
    
    "Borderline": "🌿 Sınır çizmek, kendine duyduğun saygının ve yaşamını sağlıklı bir şekilde yönlendirme arzusunun güçlü bir ifadesidir. Kimi zaman hayır demek zor gelir, suçluluk hissi sarar insanı… ama bil ki herkesin duygusal, fiziksel ve zihinsel alanlarını korumaya hakkı vardır.Sınırlar, insanlara mesafe koymak için değil; ilişkilere daha sağlıklı bir zemin oluşturmak içindir. Ne hissettiğini söylemek, neye ihtiyacın olduğunu açıkça ifade etmek, hem kendini hem de karşındakini daha iyi tanımana yardımcı olur. Senin sınırların değerlidir ve onların farkında olmak, kendine verdiğin önemin bir göstergesidir.",
    
    "Psikosomatik Bozukluklar": "🧠 Psikosomatik bozukluklar, zihinsel ve duygusal sıkıntıların bedende fiziksel belirtilerle kendini göstermesidir. Baş ağrısı, mide sorunları, kas ağrıları gibi şikayetler, altta yatan stres, kaygı veya bastırılmış duyguların bir yansıması olabilir. Kimi zaman kişi bu belirtileri gerçek bir hastalık gibi yaşar, ama tıbbi olarak bir neden bulunamayabilir. Bu durum seni çaresiz hissettirebilir ama unutma: bu belirtiler gerçektir ve yardım alarak iyileştirilebilir.✨ Terapi süreci, iç dünyandaki dinamikleri keşfetmeni ve beden-zihin arasındaki bağı anlamanı sağlar. Psikolojik destekle birlikte bedenin de zamanla rahatlayabilir. Kendini suçlama—bu yaşadıkların seni değil, yardım almayı hak eden biri olduğunu gösterir.",
    
    "Obsesif Kompulsif Bozukluk": "🔄 Obsesif-Kompulsif Bozukluk (OKB), kişinin istemsiz şekilde zihnine üşüşen düşüncelerle baş etmeye çalışırken belirli davranışlara yönelmesine neden olabilir. “Ya kapıyı kilitlemediysem?” gibi takıntılı düşünceler, sık sık kontrol etme, temizlik yapma ya da düzen kurma gibi tekrar eden eylemlerle kendini gösterebilir. Bu durum kişinin yaşam kalitesini ciddi şekilde etkileyebilir—ama unutma, OKB ile başa çıkmak mümkün.🧩 Bilişsel davranışçı terapi (BDT), maruz bırakma ve tepkiyi engelleme (ERP) gibi yöntemler, bu döngüleri kırmaya yardımcı olabilir. Bir uzmandan destek almak, bu takıntıların kaynağını anlaman ve onların üzerindeki kontrolünü yeniden kazanman için en etkili adımlardan biridir. Yardım istemek cesaret ister, ama bu cesaret aynı zamanda iyileşmenin başlangıcıdır.",
    
    "Çocuklarda Davranış Bozuklukları": "🧸 Çocuklarda davranış bozuklukları, hem çocuk hem de ailesi için yıpratıcı bir süreç olabilir. Öfke patlamaları, kurallara uymama, saldırganlık ya da içe kapanma gibi belirtiler, çoğu zaman çocuğun iç dünyasında yaşadığı duygusal çatışmaların dışa vurumudur. Bu durum cezayla değil, anlayışla ele alınmalıdır.Ebeveyn olarak sabırlı olmak, çocuğun duygularını ciddiye almak ve güvenli bir iletişim ortamı oluşturmak büyük fark yaratabilir. Bu süreçte bir çocuk psikoloğundan veya pedagogdan destek almak, çocuğun davranışlarının altında yatan nedenleri keşfetmek ve ona uygun gelişimsel yaklaşımı belirlemek için çok kıymetlidir.🪁 Unutma, zorlayıcı davranışlar bir yardım çağrısı olabilir. Çocuklar sevildiklerini ve anlaşıldıklarını hissettikçe davranışları da olumlu yönde değişmeye başlar.",
    
    "Depresyon ve İlişkili Bozukluklar": "🌧 Depresyon, insanın yaşam enerjisini adeta gölgeleyen, derin bir boşluk ve çaresizlik hissiyle kendini gösteren bir ruhsal durumdur. İlgili bozukluklar—distimi, mevsimsel depresyon, bipolar bozukluk gibi durumlar—kişinin duygularını, düşünce biçimini ve günlük işlevselliğini ciddi şekilde etkileyebilir. Bu durum çoğu zaman dışarıdan anlaşılması zor olan içsel bir mücadeledir, ama unutma: bu zorluğu yalnız yaşamak zorunda değilsin.🪷 Profesyonel destek almak, depresyonun nedenlerini anlamanda, düşünsel kalıplarını yeniden yapılandırmanda ve duygusal yükünü hafifletmende büyük fark yaratabilir. Terapi, psikiyatri desteği ve bazen ilaç tedavisi; hepsi bu sürecin iyileştirici araçları olabilir. Hissettiklerin geçerli, değerlisin ve bu desteği sonuna kadar hak ediyorsun.",
    
    "Aile ve İlişki Sorunları": "💔 Aile ve ilişki sorunları, insanın en yakınında olanlarla yaşadığı çatışmalar nedeniyle en derinden etkileyebilir. Sürekli tartışmalar, anlaşılmama hissi, sınırların ihlal edilmesi veya duygusal kopukluklar; hepsi bireyin ruhsal sağlığını sarsabilir. Ama unutma buse: ilişkiler sadece sorunlarla değil, çözüm yollarıyla da şekillenir.🛤 Bu durumları anlamlandırmak, duygularını ifade edebilmek ve sınırlarını netleştirebilmek için bir uzmandan destek almak büyük fark yaratabilir. Aile terapisi, çift danışmanlığı veya bireysel terapi süreçleri, ilişkileri yeniden yapılandırmak ve duygusal ihtiyaçları karşılamak için etkili araçlardır.",
    
    "Dikkat Eksikliği ve Hiperaktivite Bozukluğu": "💪 Spor psikolojisiyle ilgili zorluklar yaşamak, başarının eşiğinde bile insanı yalnız hissettirebilir. Müsabaka öncesi yükselen kaygı, motivasyon eksikliği ya da hataya karşı aşırı hassasiyet, zihinsel gücü zayıflatabilir. Ama bilin ki buse: bu sorunlar yalnızca senin değil, birçok sporcunun karşılaştığı ortak mücadelelerdir. Ve çözüm, yalnızca performansta değil, duygusal dayanıklılıkta saklıdır.🎯 Zihinsel antrenmanlar, destekleyici koçluk ve spor psikolojisi uzmanları; bu süreci kolaylaştıran güçlü kaynaklardır. Kendini anlamak, güçlü ve zayıf yanlarını fark etmek, başarıya giden yolda en değerli adımlardan biridir. Spor, sadece fiziksel değil, zihinsel bir sanat—ve sen bu sanatın başrolündesin.",
    
    "Travma": "🕊 Bir travma yaşamak, insanın dünyaya olan güvenini sarsabilir ve bazen en güvende hissettiği yerlerde bile huzursuzluk yaratabilir. Ancak yaşadığın şeyin senin suçun olmadığını ve hissettiklerinin tamamen geçerli olduğunu bilmelisin. Travma sonrası iyileşme bir süreçtir; inişli çıkışlı olabilir ama her küçük adım, yeniden güç kazanmanın bir parçasıdır. Kendine şefkatle yaklaşmak, duygularını bastırmadan ifade edebilmek ve güvenli alanlarda paylaşımda bulunmak bu süreci kolaylaştırabilir. Unutma, yaraların seni tanımlamaz; içindeki iyileşme gücü, sandığından çok daha derin ve güçlüdür. Yardım istemek bir zayıflık değil, cesaretin en açık halidir."
}

def kategori_tahmin_et(metin):
    inputs = tokenizer(metin, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    tahmin_id = torch.argmax(outputs.logits, dim=1).item()
    kategori = label_encoder.inverse_transform([tahmin_id])[0]
    mesaj = destek_mesajlari.get(kategori, "Destek mesajı bulunamadı.")
    return kategori, mesaj

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sikayet = request.form.get("sikayet", "")
        if sikayet:
            kategori, mesaj = kategori_tahmin_et(sikayet)
            return render_template("index.html", kategori=kategori, mesaj=mesaj, sikayet=sikayet)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
