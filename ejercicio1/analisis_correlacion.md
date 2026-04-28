# Nota — Decisión de exclusión de features por correlación

Análisis sobre si conviene excluir `timestamp`, `device_screen_resolution` y `time_since_last_login_s` del entrenamiento del perceptrón simple (Ej 1), basándose en su bajo coeficiente de correlación con el target.

## Tres tests independientes, mismo resultado

| Feature | Pearson | Spearman | Spread media-target por decil |
|---|---:|---:|---:|
| `timestamp` | +0.001 | +0.004 | 0.053 |
| `device_screen_resolution` | +0.025 | +0.029 | 0.051 |
| `time_since_last_login_s` | +0.002 | +0.001 | 0.021 |

- **Pearson ≈ 0** → no hay relación **lineal**.
- **Spearman ≈ 0** → tampoco hay relación **monótona** (no es que sea curva pero al menos creciente — directamente no hay tendencia ordinal).
- **Media del target por decil** ≈ constante alrededor de 0.42 (la media global del target). Spread máximo de 5% — dentro del ruido estadístico esperable para ~750 muestras por decil.

Si las features tuvieran señal monótona escondida, Spearman la detectaría. Si tuvieran cualquier estructura útil (incluso no monótona), las medias por decil oscilarían más allá del ruido. Ninguno de los tres tests muestra eso.

## Conclusión: sí, podés excluirlas para el perceptrón simple del Ej 1

Razones, en orden de importancia:

1. **No aportan información**. Los tres tests confirman cero relación con el target.
2. **Magnitudes problemáticas**. `timestamp` está en el orden de 1.7e9 y `device_screen_resolution` en 1e6. Si se meten sin normalizar correctamente, dominan el cálculo del gradiente: aunque `account_age_days` sea 100× más informativa, su gradiente queda enterrado por features que pesan 10 millones de veces más en magnitud.
3. **Menos features → entrenamiento más rápido y modelo más explicable** en el informe.
4. **Reduce riesgo de overfit** sobre 7500 filas. Con features ruidosas, el perceptrón puede memorizar patrones espurios.

## Salvedad importante para Ej 2/3 (MLP)

Esta decisión vale **solo para el perceptrón simple** (lineal o no-lineal con tanh/sigmoide). En un perceptrón simple no hay capa oculta, así que no hay forma geométrica de que estas features aporten — ni siquiera en interacción con otras.

**Para el MLP del Ej 2/3 la decisión sería distinta**: en una red con capas ocultas, dos features individualmente con `r ≈ 0` pueden tener una relación multiplicativa (`y = x₁ · x₂`) que solo emerge al combinarlas. Ahí conviene **no descartar ciegamente por `r ≈ 0`** — hay que probar empíricamente.

## Recomendación: ablation explícito en el informe

Para que la decisión quede sólida y no parezca arbitraria, conviene hacer un **ablation chico** al entrenar el perceptrón:

1. Entrenar con las **9 features** originales (todas).
2. Entrenar con las **6 features informativas** (excluyendo las tres mencionadas).
3. Comparar MSE final y curva de convergencia.

Si el MSE no empeora (o mejora) al sacar las 3 features ruidosas, queda demostrado empíricamente que estaban estorbando. Una gráfica con las dos curvas de MSE alcanza para justificar la decisión en el informe.

## Features que sí aportan (recordatorio)

| Feature | Pearson r vs target |
|---|---:|
| `account_age_days` | −0.585 |
| `quantity_purchased` | +0.563 |
| `amount_usd` | +0.557 |
| `session_duration_seconds` | −0.514 |
| `days_since_last_purchase` | −0.404 |
| `items_viewed_before_purchase` | +0.334 |

Estas seis son las que el perceptrón simple debería usar.
