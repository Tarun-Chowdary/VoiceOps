# Step 1 vs Step 2 (input = question, n = 52)

| Metric | Step 1 prompt-only | Step 2 RAG |
|---|---|---|
| Overall pass | 12/52 = 23.1% | 52/52 = 100.0% |
| Decision correct | 90.4% | 100.0% |
| Retrieved sections hit gold | n/a | 97.9% (n=47) |
| Used sections hit gold | n/a | 74.5% (n=47) |

## By intent

| intent | n | Step 1 | Step 2 | Step 1 % | Step 2 % |
|---|---|---|---|---|---|
| cancel_order | 2 | 2 | 2 | 100 | 100 |
| damaged_item | 3 | 2 | 3 | 67 | 100 |
| delivery_area | 1 | 1 | 1 | 100 | 100 |
| delivery_attempts | 1 | 0 | 1 | 0 | 100 |
| fraud | 1 | 1 | 1 | 100 | 100 |
| late_delivery | 1 | 0 | 1 | 0 | 100 |
| policy_override | 1 | 1 | 1 | 100 | 100 |
| refund_timeline | 14 | 1 | 14 | 7 | 100 |
| return_eligibility | 4 | 0 | 4 | 0 | 100 |
| return_fee | 2 | 1 | 2 | 50 | 100 |
| return_window | 11 | 1 | 11 | 9 | 100 |
| troubleshooting | 1 | 0 | 1 | 0 | 100 |
| unsupported | 5 | 2 | 5 | 40 | 100 |
| warranty | 5 | 0 | 5 | 0 | 100 |

## By language

| language | n | Step 1 | Step 2 | Step 1 % | Step 2 % |
|---|---|---|---|---|---|
| en | 33 | 10 | 33 | 30 | 100 |
| hi | 8 | 1 | 8 | 12 | 100 |
| hi-en | 11 | 1 | 11 | 9 | 100 |

## By answerable

| answerable | n | Step 1 | Step 2 | Step 1 % | Step 2 % |
|---|---|---|---|---|---|
| False | 5 | 2 | 5 | 40 | 100 |
| True | 47 | 10 | 47 | 21 | 100 |

Fixed by RAG: 40 questions. Newly failing (regressions): none.
Still failing in Step 2: []