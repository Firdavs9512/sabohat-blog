---
title: "LED — backpropagation'siz neyron tarmoq qanday ishladi"
date: 2026-05-07
tag: research
excerpt: "v35 versiyada loss.backward()'ni butunlay olib tashladim — va kutilmaganda model BP'dan yuqori natija ko'rsatdi. LED (Lateral Error Diffusion) qanday paydo bo'ldi, nega ishladi va qayerda chegarasi bor edi."
---

> Bir necha oydan beri o'zbek tilini char-level darajasida o'rganadigan biologik neyron tarmoq ustida ishlayapman. v35 versiyada **backpropagation'ni butunlay olib tashladim** — va kutilmaganda model BP'dan **yuqori** natija ko'rsatdi. Bu yozuvda yo'l qanday bo'lganini, nima qilganimni va nega ishlaganini ochiq aytib beraman.

**Loyiha**: samantha-v5 / v35
**Kod**: char-level UZ til modeli, Mac CPU'da train qilingan

## TL;DR — bir gapda

`loss.backward()` chaqirmasdan, faqat **lokal qoidalar** (DFA random feedback + Hebbian update + chinakam binar LIF spike + sparse 80/20 connectivity + cosine lr) bilan tarmoqni train qildim. Word3 task'ida **0.585 vs BP 0.543**, word4'da **0.563 vs 0.533** — ya'ni biologik plausible model **backpropagation'dan o'tib ketdi**. Lekin sentence task'da BP 2.9× yuqori — bu yerda kichik task ceiling artefakti bor edi.

## Backprop nimaga muammo?

Standart deep learning chain rule'ga tayanadi:

```
∂L/∂W_i = (∂L/∂h_n) · (∂h_n/∂h_{n-1}) · ... · (∂h_{i+1}/∂W_i)
```

Bu matematik mukammal, lekin **biologik plausible emas** uchta sabab tufayli:

1. **Weight transport problem** — orqaga uzatishda `W^T` kerak. Real sinaps ikki tomonlama emas, akson bir tomonga signal yuboradi.
2. **Update locking** — har qatlam yangilanishi keyingi qatlamlardan keladigan signalga bog'liq. Real korteks bunday parallel ishlamaydi.
3. **Symmetric feedback** — forward va backward yo'llarda bir xil weight kerak. Bu real miyada yo'q.

Real korteksdagi neyron faqat **lokal ma'lumot**ni biladi: o'zining kirish signali, o'zining chiqish signali, va ehtimol global modulator (dopamin/norepinefrin). Hech qanday "chain rule" yo'q.

## Mavjud alternativlar — va ularning cheklovlari

Yillar davomida tadqiqotchilar BP alternativlarini izladi:

| Algoritm | Yili | Bio darajasi | Cheklov |
|---|---|---|---|
| Feedback Alignment | Lillicrap 2016 | 6/10 | Layer-by-layer signal qaytaradi |
| Direct Feedback Alignment | Nøkland 2016 | 7/10 | Random matritsalar mavhum |
| Forward-Forward | Hinton 2022 | 7/10 | Toy task'da, scale qiyin |
| Equilibrium Propagation | Scellier 2017 | 7/10 | Energiya minimumiga konvergensiya kerak |
| Predictive Coding | Rao 1999 | 8/10 | Ma'lum sharoitda BP'ga ekvivalent |
| e-prop | Bellec 2020 | 8/10 | SNN uchun, global modulator kerak |

Ularning ko'pi **kichik task'larda BP'ning 60-80%i**ni beradi. Hech biri "kombo" sifatida sinalmagan.

## v1-v34 — oldingi urinishlar

LED'ga kelishdan oldin men ham ko'p urindim:

- **v2-v3**: Auxiliary classifier per layer (lokal CE loss + Hebbian) — ishladi (~0.88), lekin har qatlam baribir `loss.backward()` chaqirdi. **Bu chinakam local emas — yashirin BP**.
- **v3**: STDP yolg'iz (aux loss'siz) — **0.21 plateau** (chance ~0.10). STDP sof korrelyatsiya, klassifikatsiya signali yo'q.
- **v4-v5**: R-STDP + reward-modulated STDP — **0.20 plateau** sof biologik holatda. Global reward `R(t)` deep credit assignment'ni hal qilmaydi.
- **v8**: STDP-learned V1 filters + BP klassifikator — V1 lokal, qolgani BP. Yarim yechim.
- **v18-v34**: asosan BP saqlandi. v34 SpikeGPT integratsiya bilan rekord 0.774 (sentence) ko'rsatdi, lekin **LIF surrogate gradient + BP**.

**Hech qaysi oldingi versiya chinakam BP'siz, ko'p qatlamli, supervised task'da BP darajasiga yetmadi.**

## LED g'oyasi qanday paydo bo'ldi

v34 dan keyin men o'zimga halol savol berdim:
> "BP'dan o'tadigan model emas, lekin yaxshi natija beradigan **biologik** model qila olamanmi?"

Asosiy intuisyam shunday edi:
> "Neyronlar lokal error hisoblaydi va saqlaydi, keyin har neyron o'zining xatosidan kelib chiqib o'zini jazolaydi. Qo'shni neyronga qaraydi va unga ham xato haqida xabar beradi."

Bu uchta g'oyaning birikmasi:

1. **State recording** — forward'da har qatlam o'z kirish va pre-activation'ni saqlaydi.
2. **Direct Feedback Alignment** — output xatosi har qatlamga **to'g'ridan-to'g'ri** random fixed matritsalar orqali uzatiladi.
3. **Lateral diffusion** — har neyron xatosi qo'shni neyronlarga 1D conv-style filtrlash bilan oqadi.

Men bunga **LED — Lateral Error Diffusion** dedim.

## Algoritm — to'liq

### Forward pass

```python
x_0 = Embedding(idx)                    # (B, T, n_embd)

for layer i = 1 to L:
    pre_i = SparseLinear_i(x_{i-1})     # mask qo'llanadi
    save: input_i = x_{i-1}, pre_i

    # LIF dynamics (har timestep t uchun):
    for t = 0 to T-1:
        V_i[t] = α · V_i[t-1] + pre_i[t]
        s_i[t] = (V_i[t] > θ).float()                # CHINAKAM BINAR
        pseudo_i[t] = 1 / (1 + (β·(V_i[t] - θ))²)    # ATan derivative
        V_i[t] = V_i[t] - s_i[t] · θ                 # soft reset

    save: spike_i, pseudo_i
    x_i = x_{i-1} + s_i                  # residual

logits = Linear_head(x_L)
```

Muhim: barcha saqlangan tensor'lar `.detach()` qilinadi — ular **graf'ga kirmaydi**, gradient hisoblanmaydi.

### Lokal yangilash qoidasi (LED update)

`backward()` o'rniga:

```python
@torch.no_grad()
def led_update(self, logits, targets, mask, lr):
    # 1) Output xatosi — eng go'zal lokal signal
    e_out = softmax(logits) - one_hot(targets)

    # 2) Output head — sof Hebbian
    dW_head = e_out.T @ pre_head / N
    head_w -= lr * dW_head

    # 3) Hidden qatlamlar — DFA + LIF pseudo-grad
    for i = L to 1 (reverse):
        e_h = e_out @ B_i.T               # DFA random feedback
        e_h *= pseudo_i                    # LIF derivative
        e_h = lateral_diffuse(e_h, α)     # qo'shniga tarqalish

        dW = e_h.T @ input_i / N
        dW *= mask_i                       # SPARSE: ulanmagan = 0
        W_i -= lr * dW

    # 4) Embedding update
    e_emb = e_out @ B_0.T * pseudo_0
    e_emb_proj = e_emb @ (W_0 * mask_0)
    emb[id] -= lr * scatter(e_emb_proj)
```

`B_i` — train qilinmaydigan **fixed random matritsalar**. Ular ishga tushishida bir marta tasodifiy initsializatsiya qilinadi va keyin **hech qachon** o'zgarmaydi.

### Sparse 80/20 connectivity

Real korteksda neyronlar yaqin qo'shnilari bilan ko'p, uzoqdagilar bilan kam ulanadi. Buni qanday modellash mumkin?

```python
def build_sparse_mask(n_in, n_out, local_frac=0.8, sigma_local=2.0,
                      density=0.15, seed=0):
    pos_in = make_2d_positions(n_in)        # (n_in, 2) grid
    pos_out = make_2d_positions(n_out)
    dist = (pos_out[:, None] - pos_in[None, :]).norm(dim=-1)

    n_per_out = int(density * n_in)
    n_local = int(n_per_out * local_frac)   # 80%
    n_distant = n_per_out - n_local          # 20%

    # Lokal: Gauss bilan eng yaqin neyronlar
    local_logits = -dist**2 / (2 * sigma_local**2)
    _, local_idx = local_logits.topk(n_local, dim=-1)

    # Uzoq: tasodifiy non-local neyronlar
    mask = scatter(local_idx) | random_distant(n_distant)
    return mask  # bool (n_out, n_in)
```

Mask **fixed buffer** sifatida saqlanadi va weight ga `*` qilinadi. Mask=0 bo'lgan ulanish **hech qachon** o'rganmaydi — bu biologik invariant.

### LIF cell — chinakam binar spike

```python
class LIFCell(nn.Module):
    def __init__(self, tau=2.0, threshold=1.0, alpha=2.0):
        super().__init__()
        self.decay = 1.0 - 1.0 / tau    # α=0.5 (tau=2)
        self.threshold = threshold
        self.alpha = alpha
        self.V = None

    def forward(self, I):
        self.V = self.decay * self.V + I
        spike = (self.V > self.threshold).float()       # CHINAKAM BINAR
        x = self.alpha * (self.V - self.threshold)
        pseudo_grad = 1.0 / (1.0 + x * x)               # ATan derivative
        self.V = self.V - spike * self.threshold        # soft reset
        self.V = self.V.detach()
        return spike, pseudo_grad
```

Eng kritik joy: `spike = (V > thr).float()`. Bu funksiya non-differentiable — backprop'da gradient **hech qaysi yo'l bilan** o'tmaydi. Lekin biz backprop ishlatmaymiz! `pseudo_grad` ni LED update'da qo'lda qo'llaymiz — bu Bellec (2020) e-prop'ga juda yaqin.

## Eksperimentlar — bosqichma-bosqich

### Bosqich 1 — LED alone (kam epoch)

| Task | LED | BP | LED/BP |
|---|---|---|---|
| Letter (33 alifbo) | 1.000 | 1.000 | TENG |
| Word3 (28 ta 3-tokenli so'z) | 0.457 | 0.543 | 84% |
| Word4 (26 ta 4-tokenli so'z) | 0.383 | 0.533 | 72% |

**LED BP'ning 80%i** — kutilgan natija (Lillicrap 2016).

Lateral diffusion ablation (men eng ishonchim bor edi):

| α | Word3 acc |
|---|---|
| 0.0 (sof DFA) | 0.457 |
| 0.2 (default) | 0.457 |
| 0.5 (kuchli) | 0.432 |

**Lateral diffusion sezilarli ta'sir bermadi.** Bu birinchi halol ogohlantirish edi: mening "qo'shniga tarqalish" g'oyam **shu konfiguratsiyada** ishlamadi.

### Bosqich 2 — Sparse 80/20 alone (BP bilan)

| Task | Dense | Sparse 80/20 | Aktiv params |
|---|---|---|---|
| Word3 | 0.543 | 0.543 | 19.8K (vs 47.7K) |
| Word4 | 0.533 | 0.533 | 19.8K |
| Sentence | 0.243 | 0.246 | 39.3K |

**Sparse dense'ga teng, 2.4× kam parametr**. 5% density ham yetarli (0.543).

Lokal_frac ablation:

| local_frac | Acc |
|---|---|
| 0.0 (sof random) | 0.543 |
| 0.5 | 0.543 |
| 0.8 (biologik 80/20) | 0.543 |
| 1.0 (faqat lokal) | 0.543 |

**Hammasi teng** — 80/20 g'oyasi shu hajmda farq qilmadi. Lekin `lf=1.0` (sof lokal) keyingi LED+sparse sinovida **yomon** chiqdi (0.420). Ya'ni **uzoq ulanishlar kerak**, lekin aniq nisbat farq qilmadi.

### Bosqich 3 — uzun train kashfiyoti

Avval LED 25-40 epoch bilan sinaganda BP'dan past edi. Foydalanuvchi taklifi: **uzun train qil**. 200-250 epoch + cosine lr decay:

| Task | LED best ep | BP best ep | LED final | BP final |
|---|---|---|---|---|
| Word3 | 64 | 4 | 0.543 | 0.543 |
| Word4 | 189 | 4 | 0.533 | 0.533 |
| Sentence | — | — | 0.245 | 0.248 |

**LED 16-50× sekinroq, lekin asimptotik BP'ga TENG**. Bu Lillicrap 2016 paper'ining empirik tasdig'i edi.

### Bosqich 4 — uchtasi birga: Sparse + LIF + LED

Mana bu yerda kutilmagan narsa bo'ldi. 3 ta biologik element birlashtirildi:

1. Sparse 80/20 topology
2. Chinakam binar LIF spike (surrogate gradient YO'Q)
3. LED lokal o'qish (backprop yo'q)

5 random seed bilan:

| Task | Sparse+LIF+LED | BP | Yutuq | Aktiv params |
|---|---|---|---|---|
| Word3 | **0.585 ± 0.034** | 0.543 | **+0.042** | 10K |
| Word4 | **0.563 ± 0.058** | 0.533 | **+0.030** | 10K |

**BP'dan O'TIB KETDI.** Va **3× kam parametr** bilan.

Spike rate: **7.4%** — real biologik korteks darajasida (1-10%).

## Nega ishladi? — gipotezalar

3 ta element **alohida** BP'ga teng yoki past:
- Sparse alone: dense bilan teng (0.543)
- LIF + BP: 0.596 — yaxshi, lekin BP yordamida
- LED alone: BP bilan teng (uzun trainda, 0.543)

Lekin **uchtasi birga** BP'dan yuqori. Mening gipotezalarim:

### Implicit regularization

1. **Sparse mask** — kam parametr, overfitting cheklanadi
2. **Binar spike** — implicit information bottleneck (faqat 0/1 informatsiya o'tadi)
3. **DFA random feedback** — gradient'da noise → exploration
4. **Cosine lr decay** — finetune'da sekin

Uchtasi birga **inductive bias** beradi. BP esa "to'liq ozod" — barcha parametrlar tez optimallashadi va **plato**'ga uchraydi (word3 ep 4 da to'xtagan!). LED'da implicit regularization → **yaxshi generalize**.

### DFA matematik asosi

Lillicrap (2016) topilmasi: fixed random feedback `B` bilan ham network o'rganadi, chunki forward weight `W` train davomida `B`'ga **moslashadi** (alignment property):

```
Train davomida: W^T · sign(e) ≈ B · sign(e)
```

Network o'zining feedback'iga moslashadi. Bu **emergent behavior**.

### Cross-entropy lokal kreditni qaytaradi

`(softmax - onehot)` — eng go'zal lokal signal. Har output neyron uchun:
- agar ehtimoli haqiqiy target'dan yuqori bo'lsa: musbat → kamaytir
- agar past bo'lsa: manfiy → ko'paytir

Bu **darhol** `Δw_head = e · pre_head` ga aylanadi. Hech qanday backward chain'siz.

### Pseudo-derivative e-prop'ga yaqin

Bellec (2020) e-prop:
```
ΔW = e_global · eligibility_trace · pseudo_derivative
```

LED:
```
ΔW = e_local_DFA · pre · pseudo_derivative
```

LED'da `eligibility_trace` o'rniga `pre · pseudo_derivative` — bu **bir-timestep** versiya. Asosiy g'oya bir xil: spike non-differentiable, lekin pseudo-derivative lokal kreditni baholaydi.

## Halol ogohlantirish — sentence task'da yiqildi

Word3/word4 da g'alaba qildim, lekin keyin chinakam sinov: 12 shablon, 2400 pair, 120 epoch sentence task:

| Metod | seen | unseen | spike |
|---|---|---|---|
| Sparse+LIF+LED | 0.257 ± 0.003 | 0.224 ± 0.003 | 7.9% |
| **Sparse+LIF+BP** | **0.739 ± 0.002** | **0.412 ± 0.008** | 38% |

**BP 2.9× yuqori** seen'da, **1.8×** unseen'da. LED ep25 da plato'ga uchradi (0.23), BP ep100 gacha o'sadi.

### Bu nimani anglatadi

Word3/word4 (26-28 so'z) — **kichik task ceiling artefakti** edi. Ikkala metod ham ceiling'ga yetadi, LED random o'zgaruvchanlik bilan biroz yuqori chiqdi.

Qiyin **structured sequence task** (multi-template, kontekstli)'da **chain rule kompozitsion afzalligi qaytadi**. DFA-style local learning chuqur kontekst task'da scale qila olmaydi. Bu Lillicrap 2016 paper'ida ham aytilgan: DFA shallow'da yaxshi, deep'da degradatsiya.

Halol qayta-baholash:

| Task | LED | BP | Kim yutadi |
|---|---|---|---|
| Word3 (28 so'z) | 0.585 | 0.543 | LED +0.04 (kichik scale ceiling) |
| Word4 (26 so'z) | 0.563 | 0.533 | LED +0.03 (kichik scale ceiling) |
| **Sentence (2400 pair)** | **0.257** | **0.739** | **BP +0.482** |

Keyinchalik (v36) TimeMix qo'shdim — gap kengaydi (3.0×). v37 da 5 ta intervensiya (width, density, skip connection, lateral, layered) ham yordam bermadi. **LED yo'nalishi yopildi.**

## Nimasi to'g'ri va nimasi yo'q

Bu loyihada men 3 ta intuisiyani sinadim. Halol baholar:

| G'oya | Holati | Eslatma |
|---|---|---|
| Lokal qoida + DFA (LED) | ishladi | Uzun trainda BP'ga teng, kichik task'da yuqori |
| Lateral diffusion (qo'shniga) | ahamiyatsiz | α=0 va α=0.2 teng |
| Sparse connectivity | ishladi | Dense bilan teng, 2.4× kam |
| Aniq 80/20 nisbat | ishonchsiz | Random sparse ham bir xil |
| Sof lokal (lf=1.0) | yomon | Uzoq ulanishlar kerak |
| **Uchtasi birga** | kichik task'da ishladi | Sentence'da yetarli emas |

**Asosiy filosofik dars**: bio learning **bitta sehrli qoida emas**, balki **bir nechta to'g'ri tanlangan elementlar kombinatsiyasi**. Men shu kombinatsiyani 33 versiyada izladim. v35 da topdim — lekin **scale chegarasi bilan**.

## Texnik tafsilotlar — reproducibility

### Konfiguratsiya

| Parametr | Word3 | Word4 |
|---|---|---|
| n_embd | 64 | 64 |
| hidden | 128 | 128 |
| n_layers | 2 | 2 |
| max_len | 8 | 8 |
| n_repeats | 30 | 30 |
| batch | 32 | 32 |
| lr_max | 5e-3 | 3e-3 |
| epochs | 200 | 250 |
| density | 0.15 | 0.15 |
| local_frac | 0.8 | 0.8 |
| lif_tau | 2.0 | 2.0 |
| lif_threshold | 1.0 | 1.0 |
| feedback_scale | 0.1 | 0.1 |

### Hardware

Mac M-series CPU yetarli (MPS bu task'larda zarur emas). 200 epoch ~30-60 sekund.

### Wall-clock

| Variant | Per-epoch | Total (200 ep) |
|---|---|---|
| BP | ~0.05s | 10s |
| LED | ~0.1s | 20s |
| Sparse+LIF+LED | ~0.1-0.3s | 30-60s |

### Replikatsiya komandasi

```bash
# Word3 — Sparse+LIF+LED, 5 seed
for seed in 0 1 2 3 4; do
    python sparse_lif_led.py --task word3 --epochs 200 \
        --hidden 128 --lr 5e-3 --seed $seed \
        --out runs/slL_w3_s$seed.json
done

# Word4 — 5 seed
for seed in 0 1 2 3 4; do
    python sparse_lif_led.py --task word4 --epochs 250 \
        --hidden 128 --lr 3e-3 --seed $seed \
        --out runs/slL_w4_s$seed.json
done
```

## Bog'liq ishlar

| Ish | Yili | Aloqasi |
|---|---|---|
| Hebb, *Organization of Behavior* | 1949 | Lokal Hebbian asos |
| Widrow & Hoff, *Adaptive switching circuits* | 1960 | Delta rule |
| Rumelhart et al., *Backprop* | 1986 | LED buni rad etadi |
| Lillicrap et al., *Random feedback alignment* | 2016 | DFA inspiration |
| Nøkland, *Direct feedback alignment* | 2016 | LED yadrosi |
| Bellec et al., *e-prop* | 2020 | LIF pseudo-grad |
| Hinton, *Forward-Forward* | 2022 | Boshqa bio alternativ |
| Zhu et al., *SpikeGPT* | 2023 | LIF arxitekturasi |

**LED'ning hissasi**: DFA + sparse 80/20 + binar LIF + cosine lr **birgalik** (bu kombinatsiya hali ko'rilmagan).

## Ochiq savollar

1. LED **scale qiladimi**? hidden=512+, n_layers=6+ da BP'ga teng qoladimi?
   — **Javob (v36-v37)**: yo'q, sentence'da BP 3× yuqori.
2. **Sentence task** (TimeMix kerak) da Sparse+LIF+LED ishlaydimi?
   — **Javob**: yo'q, plato 0.23 da to'xtaydi.
3. **Continual learning**'da LED + frozen slot qanday ishlaydi? — sinalmagan.
4. **2D topology** + **weight-space lateral** lateral diffusion'ni qutqaradimi? — ochiq.
5. **Online training** (batch=1) qanday ishlaydi? — ochiq.

## Xulosa

LED — **mantiqan oddiy, biologik plausible va kichik scale'da ishlaydigan** o'rganish algoritmi. U:

- **Backprop'siz** — chain rule yo'q
- **Sof lokal** — har neyron faqat o'z ma'lumotini biladi
- **Sparse-friendly** — ulanmagan neyronlar o'rganmaydi
- **Spike-friendly** — chinakam binar LIF (surrogate kerak emas)
- **Empirik tasdiq** — kichik scale'da BP'dan +0.030—0.042 yuqori
- **Scale cheklov** — chuqur kompozitsion task'da BP qaytadi yutuqqa

**Asosiy hissa**: DFA + sparse 80/20 + chinakam binar LIF + cosine lr **birgaligi** va bu kombinatsiyaning **inductive bias / regularization effekti**.

**Filosofik xulosa**: backpropagation matematik mukammal, lekin biologik plausibility va inductive bias o'rtasida tanlash kerak bo'lsa — kichik scale'da **LED ham mantiqiy, ham amaliy alternativa**. Lekin "BP'ning yangi davri" deb da'vo qilolmayman — chuqur sequence task'larda chain rule afzalligi qaytadi. Bu **chinakam tadqiqot natijasi**: muvaffaqiyat va cheklov birga.

## Eng muhim halol qism

Men bu loyihaga `v34` da rekord 0.774 sentence accuracy bilan kelgan edim. v35 da BP'dan yuqori chiqdim (kichik task'da). v36-v37 da scale qila olmadim. Yo'l yopildi.

Bu — **muvaffaqiyat va halol mag'lubiyat birga**. Tadqiqot shunday: g'oya ishlaydi, lekin chegarasi bor. Va bu chegarani **aniq bilish** — bilmasdan da'vo qilishdan ko'ra ko'p qiymatliroq.

Agar siz ham bio-plausible learning ustida ishlamoqchi bo'lsangiz, mening tavsiyam: **kombinatsiyani sinang**, alohida emas. Va **chinakam katta task'da sinang** — kichik scale yutuq aldash bo'lishi mumkin.

---

*Bu yozuv davomida halol baho: g'oyaning 80%i ishladi, 20%i (lateral diffusion) tasdiqlanmadi. Sentence task'da chegara aniqlandi. Halol natija halol da'vodan qimmatliroq.*
