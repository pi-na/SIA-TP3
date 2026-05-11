### Resultados — Best combo del grid de regularización

**Best combo (paso 2):** L2 = `0.001` · σ = `0` · val_acc CV = **0.9750 ± 0.0018** · gap = **0.0750**.

**Generalización externa (test):**

![[best_reg_test_confusion_matrix.png]]

| Métrica | Test (best_reg) |
| --- | --- |
| accuracy        | **0.9601 ± 0.0030** |
| macro_precision | 0.9605 ± 0.0029 |
| macro_recall    | 0.9591 ± 0.0030 |
| macro_F1        | **0.9594 ± 0.0030** |

**Métricas por clase en test:**

| clase | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| 0 | 0.966 | 0.990 | 0.978 | 245 |
| 1 | 0.976 | 0.991 | 0.983 | 282 |
| 2 | 0.963 | 0.965 | 0.964 | 258 |
| 3 | 0.923 | 0.988 | 0.955 | 252 |
| 4 | 0.969 | 0.969 | 0.969 | 245 |
| 5 | 0.961 | 0.910 | 0.935 | 222 |
| 6 | 0.962 | 0.974 | 0.967 | 239 |
| 7 | 0.968 | 0.948 | 0.958 | 257 |
| **8** | **0.981** | **0.904** | **0.941** | 243 |
| 9 | 0.938 | 0.952 | 0.945 | 252 |

