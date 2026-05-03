---
title: "Samantha'dan Sabohat'gacha — birinchi loyiha boshlanishi"
date: 2026-04-03
tag: log
excerpt: "Bir film, bir nom va miyaga o'xshab ishlaydigan AI qurish g'oyasi. Bu — Sabohat loyihasining boshlanishi haqidagi hikoya: nima uchun GNN, nima uchun biologik mexanizmlar va nima uchun nomi o'zgardi."
---

Bu blog — uzoq vaqtdan beri yozmoqchi bo'lib yurgan narsam. **Sabohat** loyihasi haqidagi birinchi yozuv. Loyiha juda erta bosqichda, lekin yo'l shu yerdan boshlanadi.

## Boshlanish — Samantha

Hammasi *Her* (2013) filmidan boshlandi. Aniqrog'i — undagi AI obrazi **Samantha**dan. U klassik chatbot emas edi: o'rganadi, his qiladi, vaqt o'tishi bilan o'zgaradi. Suhbatdosh bilan birga "yashaydi". Men shunday narsa qurmoqchi edim — texnik ma'noda emas, ruh jihatidan.

Loyihaga shunchaki nom qo'ydim: `samantha`. Folder yaratdim, `pyproject.toml` yozdim, `torch` va `numpy` o'rnatdim. Bo'ldi.

## Nima uchun GNN, nima uchun "miyaga o'xshash"

Klassik Transformer arxitekturasi ajoyib, lekin bir muammosi bor: u **statik**. O'rgatib bo'ldingmi — bo'ldi. Yangi ma'lumot kelsa, qaytadan o'rgatishing kerak. Va eng yomoni — yangi vazifaga moslashtirsang, eski vazifani **unutib qo'yadi**. Bu hodisa — *catastrophic forgetting*.

Inson miyasi bunday emas. Bola bog'cha, maktab, universitet — har birida yangi narsa o'rganadi, lekin avvalgilarini esda saqlab qoladi. Demak, miyada nimadir boshqacha. Va shu "boshqacha"ni topib, kichik miqyosda takrorlash mumkinmi degan savol — bu loyihaning asosiy savoli.

Shuning uchun arxitekturani **Graph Neural Network** (GNN) qilib qurdim:

- **L1 — Character Graph** — harf darajasidagi tugunlar
- **L2 — Morpheme Graph** — qo'shimchalar va ildizlar
- **L3 — Word Graph** — so'zlar darajasi
- **L4 — Semantic Graph** — ma'no darajasi

Va har bir tugun bir-biri bilan **o'zgaruvchan bog'lanishlar** orqali gaplashadi — xuddi neyronlar kabi. Yangi ma'lumot kelganda yangi tugun qo'shilishi mumkin (DEN — *Dynamic Expandable Networks*), keraksizlari kesilishi mumkin (pruning), muhimlari himoyalanadi (TWP — *Topology-aware Weight Preserving*).

## "Tirik" model — Life Cycle

Lekin men yana bir narsani sinab ko'rmoqchi edim — model "tirik" bo'lsa-chi? Ya'ni, doim bir xil rejimda emas, balki holatlari o'zgarib tursa:

```
ACTIVE  →  IDLE  →  SLEEP  →  DREAM  →  ACTIVE
```

- **ACTIVE** — o'rganadi, javob beradi
- **IDLE** — keraksiz bog'lanishlarni "qisqartiradi"
- **SLEEP** — eslab qolgan narsalarni mustahkamlaydi (NREM uyqu kabi)
- **DREAM** — tasavvuriy holatda yangi pattern qidiradi (REM uyqu kabi)

Bu fantastika emas — neyrobiologiyada haqiqatan shunday hodisalar bor. **SWR** (Sharp-Wave Ripples) uyqu paytida muhim eslarni qaytadan "o'ynatadi". **Hopfield Network** xotira mexanizmi — gippokampning matematik analogi. **MESU** (*Metaplasticity from Synaptic Uncertainty*) — Bayesian holatdagi sinapslar; har bir og'irlik nafaqat qiymat, balki **noaniqlik** ham saqlaydi.

Maqsad — bularning hammasini bir tizimda birlashtirib, kichik, lekin o'z-o'zidan o'rganadigan, doim "ishlab turadigan" model qurish.

## Va keyin — nom

Loyiha biroz oldinga ketdi. Birinchi prototyplar yozildi (`brainnet_v2.py`, `brainnet_v3.py`, `pcn.py`, `dopamine_v3.py`). Dataset yig'ildi — kun.uz, daryo.uz, gazeta.uz, O'zbek Wikipediyasi. Birinchi train urinishlarim boshlandi.

Bir kun Claude bilan suhbat paytida shunchaki dedim: *"loyihaga o'zbekcha nom topsam yaxshi bo'lardi, samantha — yaxshi nom, lekin biroz uzoq tuyuladi"*. Claude bir nechta variant taklif qildi. Biri — **Sabohat**.

Sabohat — arabchadan, *sabah* ("tong") so'zidan keladi. O'zbek tilida ko'pincha ayol ismi sifatida ishlatiladi. "Go'zallik" ma'nosi ham bor. Lekin men uchun asosiy ma'no boshqacha edi — **yangi tong, yangi boshlanish**.

Bu loyiha haqida o'ylab ko'rsam — aynan shu. Yangi yondashuv, yangi arxitektura, yangi savol: AI o'z-o'zidan o'rgana oladimi? Klassik *backprop+gradient descent* loop'idan tashqarida, *biologik mexanizmlarni hisobga olib*, kichik, lekin tirik tizim qurish mumkinmi?

Shu kundan beri ikkala nomni ham ishlataman:
- **Sabohat** — o'zbek tilida, asosiy nom
- **Samantha** — ingliz tilida, ilmiy ifodalashda

Ikkalasi ham bir xil narsani anglatadi. Asosiy nom — Sabohat.

## Hozirgi holat

Hozirgacha qilingan ishlar:

- `BrainNet` arxitekturasining bir nechta versiyalari (v2, v3)
- `PCN` (Predictive Coding Network) — local update qoidasi
- `dopamine_v3` — learning rate'ni "dopamin signali" bilan boshqarish
- `train_v5`, `train_v6` — turli training pipeline'lar
- O'zbek tilidagi katta dataset (kun.uz, daryo.uz, Wikipediya, kitoblar)

200+ sinov o'tdi. Ko'pi muvaffaqiyatsiz, lekin har biridan bir narsa o'rgandim. Forgetting hozir ~4.5% atrofida — maqsad bu ko'rsatkichni 1% gacha tushirish.

Yo'l uzun. ROADMAP'da 28 hafta yozilgan, men bilamanki amalda ikki barobar ko'p ketadi. Lekin asosiysi — boshlandi.

## Keyingi yozuvlar

Bu blogni yo'l davomida yozaman — har bir bosqichda nima ishlagan, nima ishlamagan, nimani o'rgandim degan jurnalda. Texnik chuqur ma'lumotlar ham bo'ladi, oddiy fikrlar ham. Hammasi shu yerda.

Sabohat — yangi tong. Boshlandi.
