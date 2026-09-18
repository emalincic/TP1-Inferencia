---
materia: Inferencia y Estimación (I206)
tp: TP1 — PCA y Simulación Monte Carlo
grupo: 2 personas
estado: guía para redactar el informe — ejercicios 1 y 3 documentados
última actualización: 2026-09-17
---

# TP1 — Documentación de trabajo (borrador para el informe)

> Guía de metodología, resultados e interpretaciones para escribir el informe. No reemplaza el informe final. Esta actualización cubre los Ejercicios 1 y 3; la sección del Ejercicio 2 se conserva como borrador y no se actualiza aquí.

**Fuentes de esta versión:** código y salidas guardadas de `TP1_Lincic_Cordon.ipynb` para el Ejercicio 1 y de `Ejercicio3.ipynb` para el Ejercicio 3. El notebook unificado tiene, por ahora, solamente el encabezado del Ejercicio 3. `Ejercicio1.ipynb` es una versión anterior: no usarlo como referencia principal para las extensiones nuevas.

Los resultados del Ejercicio 1 se transcriben de las salidas existentes, sin volver a entrenar los 200 modelos. La simulación sintética del Ejercicio 3 se comprobó por separado, siguiendo el orden de generación aleatoria de su código actual. No se modificaron los notebooks.

## 1. Contexto del TP

- **Dataset**: PneumoniaMNIST — radiografías de tórax, 128×128 px, grises. 2 clases: 0 = Normal, 1 = Neumonía.
- **Tamaño**: 2428 imágenes de train, 468 de test (`dataset_tp1/train_labels.csv`, `test_labels.csv`).
- **Ejercicio 1**: baseline LR sin PCA, PCA parametrizado por K, scatter de las 2 primeras componentes y curva accuracy vs. K. Extensiones propias: reconstrucciones con MSE, matriz de confusión, ejemplos de errores y rechazo de entradas atípicas por distancia y reconstrucción.
- **Ejercicio 2**: Monte Carlo sobre PCA(K=2)+LR ya entrenado, perturbando test con rotación 180° con probabilidad p. Estimar E[A_p], P(L(p)>δ=0.1), histogramas de accuracy por p.
- **Ejercicio 3**: justificación teórica vía Ley de los Grandes Números del estimador de probabilidad usado en 2(c). Extensiones propias: trayectoria de un estimador con parámetro conocido y distribución del estimador al aumentar el número de simulaciones.
- **Entrega**: ZIP `TP1_GXX.zip` con informe PDF (`TP1_GXX.pdf`) + notebook(s). No incluir la carpeta de imágenes.
- **Fecha de entrega y nombres definitivos de archivos**: confirmar en el campus. No usar como referencia la estimación de fechas que figuraba en el borrador anterior.

## 2. Alcance y organización del informe

Esta guía desarrolla los Ejercicios 1 y 3. Se retiran los acuerdos de reparto del borrador anterior porque ya no reflejan el trabajo actual. La coordinación y redacción del Ejercicio 2 quedan a cargo del grupo.

Tomando como referencia la organización del TP modelo, el informe puede dividirse en:

1. **Resumen:** problema, métodos principales y resultados centrales; redactarlo al final.
2. **Introducción:** contexto del problema, objetivo de PCA y de Monte Carlo. Explicar que reducción de dimensión, clasificación y reconstrucción son objetivos distintos.
3. **Metodología:** datos y separación train/test; regresión logística; PCA por SVD; diseño de las simulaciones; extensiones propias y sus criterios.
4. **Resultados:** subsecciones por ejercicio. Cada figura debe tener una explicación que responda qué se observa y qué permite concluir.
5. **Conclusiones:** respuesta a las consignas, aportes propios y limitaciones.
6. **Referencias:** enunciado, material teórico y fuentes efectivamente utilizadas.

Las extensiones deben presentarse como análisis adicionales, no como requisitos del enunciado. No hace falta agregar un espectro de autovalores ni porcentajes de varianza explicada a los ejes para cerrar este trabajo.

## 3. Ejercicio 1 — Clasificación y reducción de dimensión

### 3.1 Datos y protocolo común

Cada imagen se aplana como un vector de $d=128\times128=16384$ intensidades. Las filas de $X_{train}$ y $X_{test}$ representan imágenes. Las etiquetas son 0 = Normal y 1 = Neumonía; el conjunto de test contiene 234 imágenes de cada clase.

Se conserva la escala de intensidad de las imágenes, sin una estandarización adicional en el código actual. La media, la base PCA, los clasificadores y los parámetros de los detectores se ajustan con train. Test se utiliza para evaluar y visualizar. Las entradas sintéticas se mantienen en arreglos separados: no se agregan al dataset ni a `test_labels.csv`.

### 3.2 Inciso (a): regresión logística sin PCA

Se entrena `LogisticRegression(max_iter=2000)` con los 16384 atributos originales y se evalúa en las 468 imágenes de test. El accuracy es la proporción de predicciones correctas:

$$\widehat A=\frac{1}{n_{test}}\sum_{j=1}^{n_{test}}\mathbb{1}\{\widehat y_j=y_j\}.$$

**Resultado guardado:** $\widehat A_{sin\ PCA}=383/468=0.818376$, es decir, **81.84%**. Este es el baseline del Ejercicio 1, no el $A_0$ del Ejercicio 2, que corresponde al modelo PCA+LR con $K=2$ sin perturbación.

### 3.3 Inciso (b): PCA parametrizado por K

Se calcula la media por píxel exclusivamente en train:

$$\mu_X=\frac{1}{n_{train}}\sum_j x_j,\qquad X_c=X_{train}-\mathbf{1}\mu_X.$$

Luego se aplica SVD reducida (`full_matrices=False`):

$$X_c=USV^T.$$

Las columnas de $V$ son las direcciones principales, ordenadas por valores singulares decrecientes. Para un $K$ elegido, $V_K$ contiene las primeras $K$ columnas y la proyección de cualquier conjunto de imágenes es

$$Y_K=(X-\mathbf{1}\mu_X)V_K.$$

La relación con la covarianza muestral es $C_X=VS^2V^T/(n_{train}-1)$: los autovalores asociados son $\lambda_i=s_i^2/(n_{train}-1)$. No hace falta construir la matriz de covarianza de $16384\times16384$ para obtener las componentes. La SVD evita ese costo y evita formar explícitamente un producto que puede empeorar el condicionamiento numérico; no significa que toda diagonalización de una covarianza sea inestable.

**Separación train/test:** para test se reutilizan exactamente $\mu_X$ y $V_K$ de train. No se ajusta un PCA independiente al test. PCA no utiliza las etiquetas para determinar sus direcciones y maximiza varianza, no separación de clases.

**Visualización:** se grafican las coordenadas PC1 y PC2 de las imágenes de test, con color por etiqueta verdadera y ejes con igual escala. Esto es una vista bidimensional: no limita la implementación a $K=2$. Se puede clasificar con más componentes y mostrar sólo las dos primeras; un gráfico 3D tampoco mostraría todas las dimensiones si $K>3$.

#### Elipses y direcciones principales por clase

En la versión actual, las medias $m_c$ y covarianzas $\Sigma_c$ de cada clase se estiman sobre **las proyecciones de train** en dos componentes. Las elipses se superponen al scatter de test. Ya no se ajustan sobre test ni se describen como elipses de $2\sigma$.

Para cada clase se dibuja la región

$$\{y:(y-m_c)^T\Sigma_c^{-1}(y-m_c)\leq9.21\}.$$

Si $\lambda_{c,i}$ son los autovalores de esa covarianza, los semiejes de la elipse tienen longitud $\sqrt{9.21\lambda_{c,i}}$ y la orientación corresponde a sus autovectores. Las flechas del código tienen longitud $2\sqrt{\lambda_{c,i}}$: muestran dirección y dispersión, pero no llegan necesariamente al borde de estas elipses.

El valor 9.21 es aproximadamente el cuantil 0.99 de una $\chi^2$ con dos grados de libertad. El nombre adecuado es **elipse de nivel de Mahalanobis con cobertura nominal del 99% bajo un modelo gaussiano**. No es un intervalo de confianza de la media ni una garantía de cobertura del 99% en nuevos datos: se estiman parámetros y la distribución real puede no ser gaussiana.

Las elipses describen la distribución de cada clase en esta proyección; **no son las fronteras de decisión de la regresión logística**. Estar lejos de ambas clases puede indicar una entrada atípica aunque LR produzca una probabilidad muy cercana a 0 o 1. Esa probabilidad no certifica que la imagen sea una radiografía.

### 3.4 Inciso (c): accuracy según K

Se prueban **todos los enteros $K=1,\ldots,200$**. La base PCA se calcula una vez; para cada $K$ se entrena una nueva regresión logística con `max_iter=2000` usando las primeras $K$ coordenadas de train y se evalúa en el mismo test. El gráfico usa eje horizontal lineal, línea del baseline, banda de ±1 error estándar y anotación del máximo observado.

| K | Accuracy | Diferencia respecto del baseline (puntos porcentuales) |
|---:|---:|---:|
| Sin PCA | 81.84% | — |
| 1 | 59.62% | −22.22 |
| 5 | 85.68% | +3.85 |
| 20 | 82.48% | +0.64 |
| 50 | 82.91% | +1.07 |
| 100 | 83.12% | +1.28 |
| 150 | 83.33% | +1.50 |
| 200 | 81.20% | −0.64 |

**Interpretación para desarrollar:**

- El máximo observado en la grilla es $K=5$, con 401 aciertos de 468, frente a 383 del baseline: 18 aciertos netos más y una diferencia de 3.85 puntos porcentuales.
- Una sola componente resulta insuficiente para este clasificador. Agregar componentes no mejora el accuracy de manera monótona.
- Cinco atributos permiten un buen resultado observado frente a los 16384 originales. Esto es reducción de la dimensión de entrada al clasificador; no demuestra que cinco componentes expliquen toda la varianza ni constituye por sí solo una tasa de compresión del sistema completo.
- El resultado con $K=200$ queda algo por debajo del baseline. No alcanza para atribuir causalmente la diferencia a sobreajuste.
- Se eligió $K=5$ mirando el test: llamarlo **mejor K observado en este test y esta grilla**, no óptimo universal ni evaluación final independiente. Para selección rigurosa se necesitaría validación dentro de train y reservar test para la evaluación final.

#### Banda de ±1 error estándar del baseline

Para un modelo fijo, suponiendo imágenes de evaluación independientes e idénticamente distribuidas, las indicadoras de acierto tienen distribución Bernoulli con probabilidad de acierto $a$. Entonces

$$\operatorname{Var}(\widehat A)=\frac{a(1-a)}{n_{test}},\qquad \widehat{SE}(\widehat A)=\sqrt{\frac{\widehat A(1-\widehat A)}{n_{test}}}.$$

Con el baseline observado, $\widehat{SE}=0.01782$, aproximadamente **1.78 puntos porcentuales**. La franja dibujada es $\widehat A_{sin\ PCA}\pm\widehat{SE}$, aproximadamente [80.06%, 83.62%]. Es una referencia de incertidumbre muestral del baseline, **no una banda de error de toda la curva** y no un intervalo de confianza al 95%.

Estar dentro o fuera de esa franja **no prueba igualdad, diferencia significativa ni equivalencia estadística**. Los modelos se evalúan sobre las mismas imágenes y sus errores son pareados; una comparación formal debería considerar ese emparejamiento, por ejemplo con McNemar o un bootstrap pareado, y la selección entre muchos K. Aquí se mantiene un análisis descriptivo.

#### Límite de K y convergencia del optimizador

El corte en 200 es una elección del experimento actual, no el máximo matemático de PCA. La matriz centrada tiene rango a lo sumo $\min(n_{train}-1,d)=2427$. La reconstrucción con $K=400$ es compatible con ello y no necesita entrenar LR.

Un aviso de máxima cantidad de iteraciones indica que el optimizador de LR no convergió con su configuración, no que PCA prohíba ese K. Antes de interpretar una accuracy con ese aviso habría que resolver la convergencia y repetir la evaluación. **No incluir la antigua caída en K=300 como resultado de la curva actual**, que sólo evalúa hasta 200.

### 3.5 Extensión: reconstrucción y fidelidad visual

Usando la misma media y base de train, se reconstruye una imagen proyectada:

$$\widetilde x_K=y_KV_K^T+\mu_X.$$

Se selecciona la primera imagen de test de cada clase y se muestran el original y las reconstrucciones con $K\in\{1,5,20,100,200,400\}$. Todas las imágenes se visualizan con escala de grises fija [0,255]. Los valores reconstruidos se recortan sólo para mostrarlos; el MSE se calcula **antes** del recorte:

$$\operatorname{MSE}_K(x)=\frac{1}{d}\sum_{j=1}^d(x_j-\widetilde x_{K,j})^2.$$

| K | MSE del ejemplo Normal | MSE del ejemplo Neumonía |
|---:|---:|---:|
| 1 | 1131 | 414 |
| 5 | 933 | 214 |
| 20 | 460 | 113 |
| 100 | 270 | 55 |
| 200 | 197 | 39 |
| 400 | 136 | 22 |

Valores redondeados de los títulos de la figura guardada; unidades: intensidad de gris al cuadrado por píxel. Con pocas componentes se conserva una estructura global suavizada; con más componentes aparecen detalles y disminuye el error.

**Conexión con la curva de accuracy:** el mejor K observado para clasificar no es necesariamente el que permite una reconstrucción visual detallada. Con una base ortonormal fija y subespacios anidados, el error de reconstrucción sin recorte no aumenta al sumar componentes; el accuracy sí puede subir o bajar. No generalizar la diferencia de MSE entre clases a partir de sólo una imagen de cada una. Esta figura no acredita preservación de detalles clínicamente relevantes.

No se ha agregado una curva de MSE promedio sobre todo test: distinguir el MSE de estos dos ejemplos de una evaluación global.

### 3.6 Extensión: matriz de confusión y ejemplos de errores

Para el modelo PCA+LR con $K=5$, la matriz guardada es la siguiente. Filas = clase verdadera; columnas = clase predicha. Se considera Neumonía como clase positiva.

| Clase verdadera / predicha | Normal | Neumonía |
|---|---:|---:|
| Normal | 192 (verdaderos negativos) | 42 (falsos positivos) |
| Neumonía | 25 (falsos negativos) | 209 (verdaderos positivos) |

Hay 401 aciertos y 67 errores, coherentes con 85.68% de accuracy. Como medidas derivadas de la misma matriz, la sensibilidad para Neumonía es $209/234=89.32\%$ y la especificidad $192/234=82.05\%$.

El código muestra los primeros dos falsos positivos y los primeros dos falsos negativos. Son ejemplos cualitativos elegidos por orden, no una muestra representativa de todos los errores. Explicar que la matriz distingue tipos de error que el accuracy resume en un solo número; en un contexto médico sus consecuencias pueden diferir. No emitir interpretaciones diagnósticas de las imágenes ni presentar este TP como validación clínica.

Estas predicciones se calculan sobre **todo test sin aplicar el filtro de rechazo**. No atribuir el accuracy de 85.68% a la extensión OOD.

### 3.7 Aporte propio: detección de entradas atípicas

**Motivación:** un clasificador binario siempre puede asignar una de sus dos etiquetas aun a entradas ajenas al problema. Una probabilidad extrema de LR no es una medida de pertenencia a la distribución de radiografías. Se explora una opción de rechazo previa o complementaria a la clasificación, sin crear una tercera clase entrenada.

#### Criterio 1: distancia de Mahalanobis en PC1–PC2

La distancia tiene en cuenta dispersión y correlación, a diferencia de la distancia euclídea. Para $y=(x-\mu_X)V_2$:

$$D_c^2(y)=(y-m_c)^T\Sigma_{c,reg}^{-1}(y-m_c),\qquad D_{min}^2(y)=\min_{c\in\{0,1\}}D_c^2(y).$$

Las medias y covarianzas por clase se ajustan con train. Para invertir la covarianza se usa $\Sigma_{c,reg}=\Sigma_c+\varepsilon_cI$, con $\varepsilon_c=10^{-6}\operatorname{tr}(\Sigma_c)/2$. Se rechaza si $D_{min}^2>9.21$, es decir, si el punto queda fuera de ambas elipses.

Aceptar por este criterio significa que la proyección es compatible con al menos una región de referencia. No significa que LR acierte, que la entrada sea necesariamente una radiografía ni que la clase más cercana deba coincidir con la predicción de LR. El 99% es nominal por clase bajo el modelo gaussiano, no la cobertura exacta de la unión de elipses.

Mahalanobis tampoco es, por sí sola, una verosimilitud: una densidad gaussiana incluye además un factor que depende del determinante de la covarianza. Por eso, comparar distancias entre clases con covarianzas diferentes no equivale necesariamente a comparar sus densidades ni sus probabilidades posteriores.

#### Criterio 2: error de reconstrucción con K=20

Una entrada extraña puede proyectarse cerca de los centros en dos dimensiones, mientras su estructura no representada por PCA sea muy diferente. Se añade el residual de reconstrucción:

$$r(x)=\frac{\|x-\widetilde x_{20}\|^2}{d}=\frac{\|x-\mu_X\|^2-\|(x-\mu_X)V_{20}\|^2}{d}.$$

La segunda expresión usa la ortonormalidad de $V_{20}$ y evita construir todas las imágenes reconstruidas. El código acota el resultado inferiormente por cero para absorber pequeños errores numéricos.

Se fija $T_r$ como el cuantil empírico 0.99 de los residuos de **train**, usando $K=20$. El valor guardado, redondeado para mostrarlo, es **$T_r\approx783.54$**; el código usa el cuantil con su precisión completa. Se rechaza si $r(x)>T_r$. K=20 es una elección exploratoria del detector, distinta de K=2 para distancia y K=5 para clasificación.

Este cuantil es un umbral de referencia aprendido, no un intervalo de confianza. Como se calcula sobre los mismos datos usados para ajustar PCA, la calibración es interna y puede resultar optimista. No garantiza rechazar exactamente el 1% de radiografías nuevas; una evaluación más rigurosa separaría un conjunto de calibración.

#### Regla combinada y experimento

Se rechaza si falla **cualquiera** de los dos criterios:

$$R(x)=\mathbb{1}\{D_{min}^2(x)>9.21\ \text{o}\ r(x)>T_r\}.$$

Se acepta únicamente si satisface ambos umbrales. El operador del código es OR (`|`), no AND. No se utiliza la etiqueta verdadera de una entrada nueva para decidir el rechazo.

Además de test, se generan 100 entradas negras, 100 blancas y 100 imágenes de ruido uniforme independiente por píxel con intensidades enteras de 0 a 255. Se usa una semilla 2026. Las 100 negras son copias de una misma imagen y las 100 blancas también: representan **dos casos únicos repetidos**, no 200 ejemplos diversos independientes.

| Tipo de entrada | Cantidad | Rechazo por distancia | Rechazo por reconstrucción | Rechazo combinado |
|---|---:|---:|---:|---:|
| Radiografías de test | 468 | 0.64% | 2.99% | 3.42% |
| Imagen negra | 100 copias | 100% | 0% | 100% |
| Imagen blanca | 100 copias | 100% | 0% | 100% |
| Ruido aleatorio | 100 | 0% | 100% | 100% |

En test se rechazan 3 imágenes por distancia, 14 por reconstrucción y 16 por la unión; una satisface ambos criterios de rechazo. Quedan aceptadas 452 de 468 (96.58%). Estas 16 son radiografías reales marcadas como atípicas, no entradas OOD conocidas.

**Hallazgo que aporta valor:** los criterios son complementarios en este experimento. La proyección bidimensional del ruido puede quedar dentro de las elipses porque descarta información en muchas direcciones; el residual detecta esa información no explicada. Las imágenes constantes se rechazan por distancia, pero no por residual: una reconstrucción de bajo error no basta para verificar que una entrada pertenece al dominio.

#### Scatter de diagnóstico: distancia frente a residual

Se representa $D_{min}^2$ en el eje horizontal y $r(x)$ en el vertical, distinguiendo radiografías, imágenes constantes y ruido. Se superponen las rectas vertical $D_{min}^2=9.21$ y horizontal $r=783.54$. En escala logarítmica:

- Abajo a la izquierda de ambos umbrales: aceptación por los dos criterios.
- A la derecha solamente: rechazo por distancia.
- Arriba solamente: rechazo por reconstrucción.
- Arriba a la derecha: rechazo por ambos.

Los umbrales delimitan regiones del detector, no fronteras de la regresión logística. Si se añaden textos para los cuadrantes, ubicarlos respecto de los umbrales y no asumir que éstos quedan en el centro del gráfico. Los valores nulos requieren un piso positivo si se muestran en ejes logarítmicos; ese piso es sólo visual.

#### Alcance y limitaciones del aporte

- Es una prueba didáctica de rechazo sobre entradas sintéticas específicas. Detectar estos casos no demuestra detectar cualquier imagen que no sea una radiografía, ni cambios de hospital, equipo o población.
- La regla reduce cobertura: también descarta radiografías válidas. Debe reportarse esa tasa junto con la detección de los casos sintéticos.
- No se midió la accuracy entre las imágenes aceptadas ni una mejora de clasificación debida al rechazo. No afirmar que el detector hace que el modelo “se confunda menos” con evidencia ya demostrada.
- No se optimizaron los umbrales mediante una validación independiente. No reajustarlos mirando test o las entradas sintéticas y luego presentar esos mismos resultados como validación independiente.
- El detector no estima la probabilidad de que una imagen sea OOD y no tiene validación clínica.

### 3.8 Figuras y tablas para llevar al informe

Orden sugerido: (1) scatter PC1–PC2 con elipses; (2) accuracy vs. K y tabla resumida; (3) reconstrucciones de ambas clases; (4) matriz de confusión; (5) ejemplos de errores; (6) tabla de rechazo y scatter distancia–residual. Las últimas cuatro son extensiones propias y pueden agruparse en una subsección o enviarse parcialmente al apéndice si el informe se hace largo.

En los pies de figura indicar: qué datos se muestran, con qué datos se ajustó el método, K utilizado, significado de las líneas/bandas y si la selección es sólo ilustrativa.

## 4. Ejercicio 2 — Simulación Monte Carlo

> Sección conservada del borrador anterior, fuera del alcance de esta actualización. Sus pendientes no implican que el notebook actual carezca de esos resultados; el grupo debe contrastarla y completarla por separado.

### 4.1 Decisiones de diseño

- Modelo PCA+LR entrenado **una única vez** sobre train sin perturbar (según aclaración del enunciado).
- Rotación 180° aplicada solo a test, en cada simulación, con probabilidad p (independiente por imagen).
- Valores de p a usar: *(pendiente de definir — enunciado pide 0 < p ≤ 0.9)*
- N_MC (número de simulaciones por p): *(pendiente de definir)*
- δ (tolerancia para la pérdida): 0.1 (fijado por el enunciado).

### 4.2 Resultados

- (a) Scatter de las 2 primeras componentes de test perturbado, por valor de p: *(pendiente)*
- (b) E[A_p] vs. p: *(pendiente)*
- (c) P(L(p) > 0.1) vs. p: *(pendiente)*
- (d) Histogramas de accuracy por p: *(pendiente)*

### 4.3 Dificultades y cómo se resolvieron

*(ir completando acá)*

## 5. Ejercicio 3 — Justificación del estimador Monte Carlo

### 5.1 Qué pide la consigna y qué probabilidad se estima

El objetivo obligatorio es justificar por la Ley de los Grandes Números (LGN) el estimador utilizado en 2(c). Las simulaciones de esta sección son un aporte ilustrativo adicional, no reemplazan esa demostración.

Para un $p$ fijo se mantiene fijo el modelo PCA+LR con $K=2$, entrenado sobre train no perturbado, y el conjunto de test. En cada realización se vuelven a sortear independientemente las rotaciones de 180° por imagen con probabilidad $p$. Si $A_0$ es su accuracy sin rotaciones y $A_i(p)$ la accuracy en la realización $i$:

$$L_i(p)=A_0-A_i(p),\qquad \delta=0.1,\qquad \theta(p)=\mathbb{P}_p\{L(p)>\delta\}.$$

La probabilidad se refiere al mecanismo aleatorio de perturbación, condicionado al modelo y dataset fijos. **No es la prevalencia de Neumonía**, no es el accuracy y no es necesariamente igual a $p$. La tolerancia 0.1 equivale a una pérdida absoluta de 10 puntos porcentuales de accuracy, no a una caída relativa del 10%.

### 5.2 Demostración mediante la Ley Fuerte de los Grandes Números

Se define una indicadora por realización:

$$Z_i=\mathbb{1}\{L_i(p)>\delta\}.$$

Para el mismo $p$, las realizaciones usan el mismo mecanismo y sorteos independientes entre simulaciones. Por tanto, $Z_1,Z_2,\ldots$ son i.i.d. Bernoulli con parámetro $\theta(p)$. Aunque distintas pérdidas puedan dar el mismo valor, eso no contradice la independencia de los sorteos.

Se tiene

$$\mathbb{E}[Z_i]=\theta(p),\qquad \operatorname{Var}(Z_i)=\theta(p)[1-\theta(p)]\leq\frac14.$$

En particular, $\mathbb{E}[|Z_i|]<\infty$, de modo que se cumplen las hipótesis de la Ley Fuerte. El promedio es exactamente el estimador del enunciado:

$$\widehat\theta_{N_{MC}}(p)=\widehat{\mathbb{P}}_p\{L(p)>\delta\}=\frac1{N_{MC}}\sum_{i=1}^{N_{MC}}Z_i.$$

Entonces

$$\widehat\theta_{N_{MC}}(p)\xrightarrow[N_{MC}\to\infty]{\mathrm{c.s.}}\theta(p).$$

Esto establece la **consistencia fuerte** del estimador para cada $p$ fijo. Si se cambian $p$ o el modelo entre realizaciones del mismo promedio, esta justificación i.i.d. ya no se aplica tal como está escrita.

### 5.3 Propiedades adicionales: insesgadez y precisión

Por independencia y linealidad de la esperanza,

$$\mathbb{E}[\widehat\theta_N]=\theta,\qquad \operatorname{Var}(\widehat\theta_N)=\frac{\theta(1-\theta)}{N},\qquad SE_N=\sqrt{\frac{\theta(1-\theta)}{N}}.$$

El estimador es insesgado para cualquier N y su MSE coincide con esa varianza. Al multiplicar N por 10, su desvío se divide por $\sqrt{10}$; para reducir a la mitad el desvío hacen falta cuatro veces más simulaciones. La escala $N^{-1/2}$ surge de esta varianza, y la aproximación normal se justifica por el Teorema Central del Límite; **la LGN por sí sola no da esa tasa ni una banda normal**.

En el experimento real del Ejercicio 2, $\theta(p)$ es desconocida. Se puede estimar el error estándar sustituyéndola por $\widehat\theta_N(p)$, pero valores observados 0 o 1 no garantizan incertidumbre nula: la fórmula plug-in puede ser engañosa en los extremos.

### 5.4 Extensión: trayectoria con parámetro conocido

Para ilustrar la convergencia se generan directamente $Z_i\sim\operatorname{Bernoulli}(\theta)$ con **$\theta=0.3$**, semilla **42** y **2000** realizaciones. No se simula una distribución Gamma ni se vuelve a ejecutar aquí el mecanismo de radiografías: se reproduce la estructura Bernoulli de las indicadoras con un parámetro artificial conocido.

Se calcula $\overline Z_n=(Z_1+\cdots+Z_n)/n$ para cada $n=1,\ldots,2000$ y se muestra junto con el valor verdadero simulado. Ejecutando el código actual en orden, el promedio final es **0.309**. Es cercano a 0.3, pero no igual; una trayectoria finita ilustra la LGN, no la demuestra.

**Lectura de la figura:** al inicio hay oscilaciones grandes; a medida que aumenta n la trayectoria se estabiliza alrededor de $\theta$. No se exige que el error disminuya en cada paso ni que el promedio quede de un solo lado del valor verdadero.

#### Cómo llamar e interpretar la banda

La banda dibujada es

$$\theta\pm1.96\sqrt{\frac{\theta(1-\theta)}{n}},$$

con extremos recortados a [0,1] para la visualización. El nombre recomendado es **banda puntual de variabilidad aproximada del 95% del estimador bajo el modelo Bernoulli**; en una leyenda breve, **banda teórica aproximada (95% puntual)**.

Está centrada en el parámetro verdadero conocido, no en el estimador observado. Para cada n fijo y suficientemente grande, la aproximación normal describe dónde caería el estimador en aproximadamente el 95% de muchas repeticiones. **No es un intervalo de confianza calculado a partir de esa muestra para estimar un parámetro desconocido**, ni una banda simultánea que garantice contener toda la trayectoria con probabilidad 95%. Para n pequeño la aproximación normal puede ser mala; recortar los extremos no la vuelve exacta.

Puede mencionarse que un intervalo de confianza responde a otra pregunta: a partir de datos observados, construir un intervalo para el parámetro desconocido. Aquí la finalidad es ilustrar la dispersión de $\overline Z_n$ cuando $\theta$ se conoce.

### 5.5 Extensión: distribución del estimador al aumentar N

Para cada $N\in\{10,100,1000\}$ se repite **3000 veces** el experimento completo. Cada repetición genera N Bernoulli independientes y produce un promedio. Se grafican histogramas de esos 3000 estimadores con una línea vertical en $\theta=0.3$ y se comparan desvíos empíricos (`ddof=1`) y teóricos.

| N | Repeticiones | Desvío empírico | Desvío teórico $\sqrt{0.3(1-0.3)/N}$ |
|---:|---:|---:|---:|
| 10 | 3000 | 0.147 | 0.145 |
| 100 | 3000 | 0.046 | 0.046 |
| 1000 | 3000 | 0.014 | 0.014 |

Los desvíos empíricos se comprobaron siguiendo el código actual: primero las 2000 realizaciones de la trayectoria, luego los histogramas, con el mismo generador avanzando. Si se vuelve a ejecutar sólo la celda de histogramas, el generador cambia de estado y los resultados pueden variar. Para reproducirlos, reiniciar y ejecutar las celdas en orden.

**Interpretación:** la dispersión cae al aumentar N y concuerda con la fórmula teórica. N es el tamaño de cada estimación; 3000 es el número de estimaciones repetidas para observar su distribución. No confundir ambas cantidades.

Con N=10 el estimador sólo toma valores múltiplos de 0.1; los huecos del histograma responden a esa discreción, no a un error de simulación. Para N grandes se observa una distribución más concentrada y aproximadamente normal, de acuerdo con el TCL. Los paneles actuales tienen rangos horizontales distintos: leer la concentración mediante las escalas y desvíos; usar un mismo rango horizontal sería una mejora opcional de comparación visual.

### 5.6 Material para llevar al informe

1. Demostración formal con indicadoras, hipótesis i.i.d. y convergencia casi segura: respuesta central al Ejercicio 3.
2. Fórmula de varianza y error estándar: explica el efecto de aumentar $N_{MC}$.
3. Figura de promedio acumulado, aclarando el carácter sintético y puntual aproximado de la banda.
4. Figura conjunta de tres histogramas con la comparación de desvíos.

La simulación con $\theta=0.3$ no calcula la probabilidad real del Ejercicio 2 ni valida el desempeño del clasificador: ilustra por qué promediar indicadoras es un procedimiento consistente.

## 6. Ideas para las conclusiones del informe

Estas son ideas para desarrollar con redacción propia, no un cierre listo para copiar:

- **Respuesta al Ejercicio 1:** PCA reduce la dimensión de entrada; en este test, cinco componentes alcanzan el mayor accuracy de la grilla (85.68%), frente a 81.84% sin PCA. La elección fue exploratoria sobre test y no establece un óptimo general.
- **Clasificación frente a reconstrucción:** agregar componentes reduce el residual, pero no asegura aumentar el accuracy. Una representación útil para clasificar no necesita reconstruir todos los detalles visibles.
- **Información más allá del accuracy:** la matriz distingue 42 falsos positivos y 25 falsos negativos. Los ejemplos permiten observar errores concretos sin sustituir una evaluación sistemática.
- **Aporte de rechazo:** distancia y residual detectan fallas diferentes; su unión rechaza todos los casos sintéticos ensayados, pero también 16 radiografías válidas. Reportar tanto detección como pérdida de cobertura.
- **Respuesta al Ejercicio 3:** el estimador de probabilidad es un promedio de indicadoras i.i.d. y converge casi seguramente por la LGN. Las simulaciones ilustran esa convergencia y la reducción del desvío en escala $N^{-1/2}$.
- **Limitaciones comunes:** un test finito, selección de K sobre ese test, calibración interna del rechazo y pruebas sintéticas simples. No extraer garantías clínicas ni significancia estadística sin análisis específico.
- **Integración pendiente:** enlazar estas conclusiones con los resultados del Ejercicio 2 una vez que su documentación esté completada.

## 7. Revisión antes de redactar y entregar

### 7.1 Detalles observados en el notebook que conviene corregir manualmente

No se corrigieron aquí porque esta tarea sólo modifica la documentación:

- En la curva de accuracy, `ax.legend("lower right")` no especifica la ubicación: debe usarse `ax.legend(loc="lower right")` para conservar las etiquetas correctas.
- La anotación del máximo combina `textcoords="offset points"` con `xytext` construido como si fueran coordenadas de datos. Elegir un desplazamiento en puntos, por ejemplo `(12, -35)`, o volver a coordenadas de datos; verificar que el texto no tape el título ni salga del panel.
- Corregir **“Normál” → “Normal”** en los gráficos.
- Si se mantienen ejes simples en el scatter como se decidió, retirar los porcentajes de varianza explicada de las etiquetas actuales. No afecta el ajuste de PCA.
- Aclarar en título o pie de las elipses que el 99% es nominal bajo una aproximación gaussiana con parámetros ajustados en train.
- Revisar que los textos de cuadrantes del scatter distancia–residual correspondan a las regiones delimitadas por los umbrales, no a posiciones fijas arbitrarias del panel.
- En `Ejercicio3.ipynb`, cambiar el comentario “Simulación que prueba esto” por “Simulación que ilustra la convergencia”. Usar para la banda una leyenda que indique aproximación y cobertura puntual.
- Incorporar las celdas del Ejercicio 3 al notebook unificado o entregar el notebook separado, según la modalidad acordada: el archivo unificado actual contiene sólo su encabezado.

### 7.2 Comprobaciones de contenido y reproducibilidad

- [ ] Reiniciar y ejecutar en orden los notebooks definitivos; verificar ausencia de errores y avisos de convergencia de LR. Esta actualización no reemplaza esa ejecución completa.
- [ ] Usar una única versión de las salidas para figuras, tablas y texto; no mezclar la antigua curva hasta K=300 ni histogramas de ejecuciones con distintos estados del generador.
- [ ] Mantener separado el baseline sin PCA del $A_0$ de PCA+LR con K=2 del Ejercicio 2.
- [ ] Identificar K=2 para distancia/visualización, K=5 para el clasificador seleccionado y K=20 para el residual del detector.
- [ ] No afirmar que las elipses sean intervalos de confianza ni que la banda ±1SE pruebe significancia.
- [ ] No afirmar que la banda de la simulación cubra toda la trayectoria con probabilidad 95%.
- [ ] Explicar que los porcentajes de rechazo se calculan sin modificar las etiquetas originales y que la matriz de confusión no incorpora el filtro OOD.
- [ ] Exportar figuras legibles, con unidades, leyendas y pies consistentes; evitar capturas que incluyan código o salidas ajenas a la figura.
- [ ] Completar la sección del Ejercicio 2 y las referencias reales antes de cerrar el informe.
- [ ] Confirmar en el campus fecha, identificación de grupo, nombres de archivos y requisitos de entrega.
