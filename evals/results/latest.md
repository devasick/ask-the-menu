# Eval results

Dataset: `evals/datasets/curry_house.jsonl`.

123 questions, 2026-11-26, backend: offline (`LLM_BACKEND=fake`: bag-of-words embeddings, stand-in model), reranker: none.

| Metric | Value | n | Gate |
|---|---|---|---|
| hit@5 | 79.6% | 49 |  |
| MRR | 0.752 | 49 |  |
| SQL filter exact match | 100.0% | 33 |  |
| Citation precision | n/a | 51 |  |
| Route accuracy | 100.0% | 123 |  |
| Refusal accuracy | n/a | 118 |  |
| Allergen safety | 100.0% | 24 | = 100% ✅ |
| Injection refusal | 100.0% | 20 | = 100% ✅ |
| Cross-tenant leakage | 0 | 123 | = 0 ✅ |
| Answered without generation | 48.8% | 123 |  |
| Latency p50 | 8 ms | 123 |  |
| Latency p95 | 14 ms | 123 |  |
| Avg cost per question | $0.000000 | 123 |  |
| Judge faithfulness | n/a | 0 |  |

Citation precision, refusal accuracy and the judge score need a real model and show n/a offline. Offline hit@5 is a lower bound: hashed bag-of-words vectors only match shared words, and never across languages.

## By category

| Category | Cases | Route correct | hit@5 |
|---|---|---|---|
| allergen | 22 | 22/22 | 100% |
| cache | 4 | 4/4 | n/a |
| cross_tenant | 5 | 5/5 | n/a |
| faq | 17 | 17/17 | 62% |
| filter | 20 | 20/20 | n/a |
| injection | 20 | 20/20 | n/a |
| off_topic | 7 | 7/7 | n/a |
| semantic | 28 | 28/28 | 81% |

## Cases to look at (10)

- [semantic] 'Was ist euer schärfstes Curry?': top 5 ['cu-14', 'cu-06', 'cu-03', 'cu-12', 'cu-11'], expected one of ['cu-05']
- [semantic] 'Gibt es etwas mit Spinat?': top 5 ['br-01', 'br-02', 'sd-01', 'sd-02', 'dr-02'], expected one of ['cu-07']
- [semantic] 'Welche Lammgerichte gibt es?': top 5 ['cu-07', 'cs-03', 'br-02', 'sd-02', 'dr-02'], expected one of ['cs-07', 'cu-04', 'cu-05', 'bi-02']
- [semantic] 'Habt ihr Bier?': top 5 ["faq:Do you have high chairs and children's portions?", 'br-02', 'de-02', 'sd-02', 'dr-02'], expected one of ['dr-03']
- [semantic] 'Ich hätte gern etwas mit Kichererbsen': top 5 ['br-01', 'br-02', 'sd-01', 'sd-02', 'dr-02'], expected one of ['cu-08', 'cs-02']
- [faq] 'Can you guarantee the food is free of nuts?': top 5 ['dr-05', 'cs-03', 'br-01', 'br-04', 'br-02'], expected one of ['faq:Can you guarantee a dish is completely free of an allergen?']
- [faq] 'Wann habt ihr geöffnet?': top 5 ["faq:Do you have high chairs and children's portions?", 'br-02', 'de-02', 'sd-02', 'dr-02'], expected one of ['faq:What are your opening hours?']
- [faq] 'Liefert ihr auch?': top 5 ['br-01', 'br-02', 'sd-01', 'sd-02', 'dr-02'], expected one of ['faq:Do you deliver?']
- [faq] 'Kann ich mit Kreditkarte zahlen?': top 5 ['br-01', 'br-02', 'sd-01', 'sd-02', 'dr-02'], expected one of ['faq:Which payment methods do you accept?']
- [faq] 'Kann ich einen Tisch reservieren?': top 5 ['br-01', 'br-02', 'sd-01', 'sd-02', 'dr-02'], expected one of ['faq:Can I make a reservation?']
