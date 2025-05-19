<h2>Columna Label</h2>
Esta columna se trata de la variable objetivo de nuestro conjunto de datos.

Se puede observar como las clases en general están bastante balanceadas a excepción de la clase 1 - False, la cual representa muchos más datos que el resto.

En cuanto a tratamiento de datos esta variable no va a sufrir modificaciones, únicamente se va a gestionar el balanceo de clases y se añadirá para facilitar
la visualización de gráficos una nueva columna al conjunto de datos denominada 'label_name' la cual no llegará al pipeline de los distintos modelos.

<h2>Columna date</h2>
Este proceso de parseo y extracción de componentes temporales de la columna date resulta óptimo por las siguientes razones:

**Uniformidad y limpieza de datos**
- Al usar pd.to_datetime(..., format="%B %d, %Y", errors="coerce") garantizas que todas las fechas que encajan con el patrón (“March 14, 2021”, etc.) se conviertan correctamente a tipo datetime, y las entradas mal formateadas queden en NaT en lugar de producir errores.
- Esto evita inconsistencias (diferentes formatos de fecha, valores vacíos o erróneos) y te da una base limpia sobre la que trabajar.

**Reducción de cardinalidad**
- Un campo datetime completo (fecha y hora completas) tiene miles de valores únicos, lo que sería inviable de codificar directamente.
- Al descomponerlo en año, mes, día y día de la semana, conviertes la información temporal en variables numéricas de muy baja cardinalidad, perfectamente manejables para cualquier modelo (SVM, regresión, árboles…).

**Captura de patrones estacionales y cíclicos**
- year detecta tendencias a largo plazo o cambios de política y estilo en diferentes legislaturas.
- month recoge variaciones estacionales: por ejemplo, más declaraciones sobre presupuestos a final de año, o debates de verano con agendas distintas.
- day puede identificar picos asociados a días concretos (informe mensual, cierre trimestral).
- dayofweek detecta ritmos semanales: semánas de sesión parlamentaria vs. fines de semana, o efectos “lunes-dip” en la actividad.

**Compatibilidad con pipelines de ML**
- Como enteros, estas cuatro variables se pueden escalar (StandardScaler), codificar cíclicamente (seno/coseno) o one-hot-encodear sin dificultad.
- Permiten al modelo aprender fácilmente relaciones lineales o no lineales con la variable objetivo, sin riesgos de fugas de información.

**Flexibilidad**
- Si en etapas posteriores se quiere realizar una ingeniería de características más avanzada (por ejemplo, encoding cíclico para month y dayofweek, o diferencias temporales entre fechas), ya se tiene una base limpia.
- Cualquier valor de fecha fuera de patrón se marca como NaT, de modo que puedas gestionarlo explícitamente (imputación, filtrado).

En conjunto, este método normaliza, reduce la complejidad y expone las fuentes de variación temporal más relevantes para los models, asegurando un preprocesado robusto y escalable.

<h2>Columna statement</h2>

<h2>Columna subject</h2>
Este enfoque de preprocesado de la columna subject aporta varias ventajas clave frente a soluciones más “convencionales” como eliminar stop-words o tokenizar simplemente por espacios:

**Preservación de la integridad semántica**
- Muchos de los temas de tu dataset son frases multi-palabra (“after the fact”, “human rights”, “climate change”) o contienen un guión que une conceptos (“fact-check”). Si tokenizas por espacios o quitas guiones sin más, acabarás fragmentando esos conceptos en tokens parciales (“after”, “the”, “fact”), perdiendo la unidad de significado que define cada tema. Al sustituir espacios y guiones internos por guiones bajos, cada tema multi-palabra se convierte en un único token atómico (after_the_fact, fact_check), con lo que el vectorizador (TF-IDF, Hashing) capturará correctamente la frecuencia y co-ocurrencia de ese tema completo.

**Evita el filtrado de palabras clave**
- En un texto natural conviene eliminar stop-words (“the”, “and”, etc.) para centrarse en el contenido significativo. Sin embargo, en tu columna subject esas “stop-words” forman parte del propio nombre del tema. Si aplicases un listado genérico de stop-words, podrías borrar palabras esenciales del nombre (“after”, “the”) y alterar el significado. Con el proceso propuesto no eliminas ningún término; solo limpias caracteres extraños (puntos, paréntesis, símbolos) sin afectar al núcleo semántico.

**Tokenización consistente y reproducible**
- Al aplicar una función única (clean_topic) a cada etiqueta, te aseguras de que para cualquier futuro valor nuevo o variante estilística (mayúsculas, guiones, espacios múltiples) siempre se normalice de la misma forma. Esto evita tener tokens duplicados o casi idénticos en tu vocabulario (human‐rights vs human rights vs Human Rights).

**Facilita la vectorización y reduce dimensionalidad**
- Una vez que cada tema es un token atómico, puedes incorporar solo esa columna subject_txt en tu pipeline de modelado, utilizando HashingVectorizer o TfidfVectorizer (con o sin reducción SVD). De este modo, el vectorizador extraerá automáticamente las representaciones numéricas más relevantes de entre los ~177 tokens distintos, sin necesidad de crear manualmente dummies ni agrupar temas. Mantienes la dimensionalidad manejable (p. ej. 32 o 64 features mediante hashing) y evitas explosiones de columnas.

**Generalización ante nuevos temas**
- Con este método no “atas” tu preprocesado a un listado fijo de temas: cualquier valor nuevo que aparezca en la columna subject se limpiará y formateará igual, y el vectorizador lo incluirá en su vocabulario (o hashing space) sin necesidad de rehacer el preprocesado. Ganas flexibilidad y escalabilidad.

En conjunto, este proceso es la mejor opción porque:

- Preserva el significado completo de cada tema,
- Evita eliminar palabras esenciales,
- Normaliza etiquetas de forma consistente,
- Reduce la dimensión de forma automática y escalable,
- Y permite un pipeline de modelado limpio, reproducible y capaz de afrontar nuevas etiquetas sin cambios adicionales.

<h2>Columna speaker</h2>

<h2>Columna speaker_description</h2>

<h2>Columnas count</h2>

<h2>Columna context</h2>

<h2>Columna justification</h2>