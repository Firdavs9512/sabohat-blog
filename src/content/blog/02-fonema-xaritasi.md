---
title: "SNN qatlamlari uchun spike encoding usulini tanlash"
date: 2026-04-28
tag: note
excerpt: "Rate coding, temporal coding va population coding — qaysi biri Sabohat arxitekturasiga mos keladi va nima uchun."
---

Spiking network'da kirish ma'lumotini spikelar ketma-ketligiga aylantirish kerak. Bunda bir nechta yondashuv bor:

- **Rate coding** — sodda, lekin axborot zichligi past.
- **Temporal coding** — spike *qachon* keladi muhim, juda samarali, lekin treningi qiyin.
- **Population coding** — bir signalni bir nechta neyron birgalikda kodlaydi, miyaga yaqin.

Sabohat uchun gibrid yondashuv tanladim: kirish qatlamida **population + temporal**, ichki qatlamlarda esa toza **temporal**. Sinov natijalari shu yo'l boshqa variantlarga nisbatan ~12% yaxshi natija berdi.
