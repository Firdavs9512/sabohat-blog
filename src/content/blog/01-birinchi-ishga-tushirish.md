---
title: "sabohat — birinchi ishga tushirish, nima ishladi va nima ishlamadi"
date: 2026-05-03
tag: log
excerpt: "SNN asosidagi modelni dastlabki dataset bilan o'rgatib ko'rdim. Spike timing nazariyasi kutilgandan yaxshi ishladi, ammo continual learning paytida eski xotira yo'qolishi (catastrophic forgetting) hali ham asosiy muammo."
---

Bugun **Sabohat**ning birinchi ishlaydigan prototypini yig'dim. Asosiy g'oya — biologik miyaga yaqinroq bo'lgan **Spiking Neural Network (SNN)** asosida o'rganadigan model qurish.

## Nima ishladi

- SNN qatlamlari spike-based ishlov beradi, energiya samaradorligi klassik ANNga nisbatan ancha yaxshi.
- Dastlabki kichik dataset (~10k sample) ustida sinov natijalari kutilganidan yuqori — accuracy ~78%.
- Spike timing dependent plasticity (STDP) qoidasi local update qilishga imkon beradi, backprop kerak emas.

## Nima ishlamadi

Asosiy muammo — **catastrophic forgetting**. Yangi vazifaga o'rgatilganda eski vazifadagi performance keskin tushib ketadi. Bu odam miyasida bo'lmaydi, demak biz hali biologik mexanizmlarning muhim qismini soddalashtirib yubordik.

## Keyingi qadam

CL (continual learning) muammosini kamaytirish uchun **synaptic consolidation** mexanizmini sinab ko'raman — muhim aloqalarni "muzlatib" qo'yish g'oyasi.
