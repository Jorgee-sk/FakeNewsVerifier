<h2>Columna Label</h2>

Esta columna se trata de la variable objetivo de nuestro conjunto de datos. Codifica la categoría de veracidad asignada por PolitiFact a cada afirmación.
Suele representarse como un entero (por ejemplo, 0="pants-on-fire", 1="false", 2="barely-true", 3="half-true", 4="mostly-true", 5=“true”).

Se puede observar como las clases en general están bastante balanceadas a excepción de la clase 1 - False, la cual representa muchos más datos que el resto.

En cuanto a tratamiento de datos esta variable no va a sufrir modificaciones, únicamente se va a gestionar el balanceo de clases y se añadirá para facilitar
la visualización de gráficos una nueva columna al conjunto de datos denominada 'label_name' la cual no llegará al pipeline de los distintos modelos.

<h2>Columna date</h2>

La columna date registra la fecha exacta en que el orador emitió la declaración que luego ha sido verificada. 
Esta información temporal no solo permite ordenar cronológicamente los datos, sino también descubrir patrones a lo largo del tiempo. 

Inicialmente en nuestro flujo de preprocesado, la columna date se transforma en primer lugar de texto a un objeto datetime de Pandas 
empleando un formato estricto (“%B %d, %Y”) y forzando los errores a NaT para capturar automáticamente aquellas fechas mal formateadas.

A partir de este objeto temporal se extraen cuatro nuevas variables —año, mes, día del mes y día de la semana—, cada una tipada como entero.

Con ello, convertimos un campo con miles de valores únicos en cuatro dimensiones manejables que permiten al modelo aprender tanto tendencias 
a largo plazo (año) como patrones estacionales (mes), cotidianos (día) y ritmos semanales (día de la semana), sin incurrir en fugas de información 
ni en explosión de categorías.

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

Esta columna contiene el texto original completo de la afirmación que está siendo verificada. 
Es la fuente primaria sobre la que se aplican todas las transformaciones de limpieza y tokenización para extraer patrones lingüísticos que distingan declaraciones verdaderas de falsas.

En la fase de preprocesado la columna statement se transforma primero convirtiendo todo el texto a minúsculas y eliminando signos de puntuación, 
números aislados y símbolos innecesarios para homogeneizar el contenido, es decir se somete a una limpieza básica del texto.

A continuación, aplicamos tokenización con NLTK para dividir cada enunciado en sus palabras, preservando negaciones y expresiones compuestas.

Tras ello, filtramos las palabras vacías (stop-words) más frecuentes en inglés —que apenas aportan valor discriminativo— y, seguidamente, lematizamos cada token con WordNet, 
reduciendo las formas flexivas y unificando variantes morfológicas.

Finalmente, los tokens resultantes se vuelven a unir en la columna 'clean_statement', generando un texto limpio y compacto.

Además cabe destacar que se añaden dos columnas extra 'statement_len_chars' y 'statement_len_words' represenando la longitud de caracteres de la cadena y la cantidad de 
palabras de la misma que aportan más información a los modelos.

Este enfoque es el más adecuado porque:

- Minimiza el ruido al eliminar puntuación y stop-words irrelevantes, evitando que el modelo aprenda de patrones innecesarios.
- Refuerza la señal lingüística con la lematización, que agrupa formas flexivas y reduce el número de valores de la matriz de términos.
- Preserva matices esenciales, como las negaciones y las entidades compuestas, que son clave para evaluar la veracidad del contenido.
- Optimiza la representación numérica con TF-IDF/BoW: un vocabulario más compacto y relevante se traduce en pesos más informativos y mejor poder discriminativo para SVM o regresión.
- Garantiza reproducibilidad y escalabilidad, pues todas las etapas son deterministas y se integran en un pipeline de scikit-learn sin riesgo de filtración de datos entre fases de entrenamiento y validación.

<h2>Columna subject</h2>

La columna subject identifica el o los temas sobre los que trata cada enunciado, tal como los categoriza el equipo de PolitiFact en la colección LIAR-2. 
Cada fila puede contener múltiples subtemas separados por “;” (por ejemplo, “economy;tax” o “health;policy”).

En el tratamiento de esta columna, esta pasa de ser un texto libre con múltiples temas concatenados a un campo estructurado y normalizado que preserva cada tópico como una unidad atómica.

Primero, dividimos la cadena original en subetiquetas separadas por “;” y las convertimos a minúsculas para un tratamiento uniforme. 
A continuación, en cada subtema reemplazamos guiones y espacios internos por guiones bajos, asegurándonos así de que cada concepto compuesto se mantenga intacto y no se fragmente durante la tokenización.

Finalmente, eliminamos cualquier carácter extraño que no aporte valor semántico (puntos, paréntesis, símbolos) y reconstruimos un mini-texto en forma de nueva variable 'subject_txt' 
donde cada token corresponde a un tema claramente delimitado. 

De este modo, dejamos lista la columna para incorporarla directamente en el pipeline de modelado con un vectorizador (TF-IDF, Hashing, etc.), 
sin necesidad de generar manualmente decenas de variables dummy ni perder información contextual.

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

En speaker aparece el nombre de la persona que pronuncia la afirmación (p. ej. “Donald Trump”, “Sen. Jane Doe”). 

Aunque en este caso no aporta directamente al modelo, esta columna sirve para agrupar o filtrar.

Dado que esta columna no aporta información directamente al modelo se va a suprimir del conjunto de datos a la hora de empezar
a tratar en el entrenamiento.

<h2>Columna speaker_description</h2>

Esta columna resume el rol o cargo del orador (por ejemplo, “45th President of the United States”, “Senator from California”).

Proporciona contexto sobre su posición institucional y autoridad, y a menudo aporta información de fondo que puede correlacionar con el estilo retórico o el nivel de precisión de la declaración.

En el flujo de preprocesado de la columna speaker_description se realiza un tratamento muy similar al aplicado a la columna 'statement'
con el fin de extraer su valor informativo de forma consistente y compacta.

Primero, convertimos todo el texto a minúsculas y eliminamos signos de puntuación, números aislados y otros símbolos que no aportan contenido semántico relevante. 
A continuación, tokenizamos la cadena para separar cada palabra o expresión clave, aplicamos un filtrado de stop-words para descartar vocabulario demasiado genérico
que apenas contribuye a discriminar estilos o roles, y luego lematizamos cada token con WordNet para unificar formas flexivas.
Por último, volvemos a unir los tokens limpios en la columna clean_speaker_description, dejando un texto coherente y libre de ruido que maximiza la señal útil.

También creamos las variables que representan la longitud de caracteres de la cadena y la cantidad de 
palabras de la misma aportando más información a los modelos.

Este enfoque es adecuado porque:

- Reduce el ruido al suprimir caracteres y palabras poco informativas, permitiendo que el modelo se concentre en títulos y cargos relevantes (“president”, “senator”, “mayor”).
- Unifica variantes léxicas (singular/plural, conjugaciones), lo que disminuye el tamaño de la matriz documento-término y favorece una representación más densa y consistente.
- Preserva las jerarquías implícitas al mantener intactas las unidades semánticas esenciales, de modo que “vice_president” y “vice_president_of_the_united_states” no se fragmenten en tokens irrelevantes.
- Facilita la vectorización mediante TF-IDF o BoW, ya que el vocabulario final es compacto, relevante y reproducible en cada fold de validación, garantizando la estabilidad del pipeline de modelado y evitando fugas de información.

<h2>Columnas count</h2>

Estas seis columnas representan el número de noticias de cada cada categoría de veracidad por speaker en el histórico del dataset. 

En nuestro preprocesado transformamos los recuentos históricos de verificaciones de cada orador en proporciones normalizadas 
(true_prop, mostly_true_prop, half_true_prop, mostly_false_prop, false_prop y pants_on_fire_prop).

Para ello, dividimos el número de veces que a un mismo hablante se le asignó cada etiqueta de veracidad por su total de comprobaciones en el dataset.
De este modo, en lugar de trabajar con valores absolutos obtenemos seis variables acotadas entre 0 y 1 que reflejan su “reputación”
o tendencia histórica a decir la verdad o a exagerar.

Este enfoque es óptimo porque:

- Normaliza para speakers con diferente volumen de datos, haciendo las proporciones comparables sin importar cuántas verificaciones tengan.
- Reduce el impacto de outliers, ya que un speaker con pocas intervenciones no domina el modelo con recuentos elevados.
- Agrega señal predictiva sobre la credibilidad previa: los modelos pueden aprender que, por ejemplo, un índice alto de false_prop suele asociarse a declaraciones más propensas a ser incorrectas, mientras que un true_prop elevado indica mayor fiabilidad.
- Facilita el aprendizaje en algoritmos lineales o basados en distancia, al trabajar con variables continuas acotadas en lugar de contar con múltiples features discretas de distinto rango y magnitud.

<h2>Columna context</h2>

En context se almacena un breve texto descriptivo del ámbito o medio en que se produjo la afirmación (por ejemplo, “white house press briefing”, “Twitter”).

Ofrece información adicional sobre el escenario y el canal, lo cual puede influir en el tono y fiabilidad de lo dicho.

La columna context se convierte en un conjunto de variables categóricas binarias que capturan de forma explícita el tipo de entorno.
Primero, aplicamos reglas de búsqueda de palabras clave las cuales agruparán en una serie de categorías:

- social_media
- advertising
- verbal_event
- document
- media
- location

Para cada una de estas categorías se creará una variable dummy a excepción de una de ellas para evitar la colinealidad perfecta. 
En este caso la dummy no representada es advertising.

Las variables serán flags que representan valores booleanos (0/1): ctx_document, ctx_location, ctx_media, ctx_social_media y ctx_verbal_event

Una vez generadas estas flags, eliminamos el texto original de context.

Este enfoque resulta óptimo porque abstrae la riqueza semántica del texto libre en indicadores claros y de muy baja cardinalidad,
elimina el ruido léxico y evita la explosión dimensional de una vectorización completa, y al mismo tiempo ofrece un punto de vista interpretable
lo que facilita que el modelo aprenda patrones de veracidad ligados al canal de comunicación sin complicar el pipeline de características.

<h2>Columna justification</h2>

La columna justification contiene la explicación y evidencias que los verificadores de PolitiFact aportan para fundamentar la categoría de veracidad asignada. 

Incluye citas, datos, estadísticas o enlaces a fuentes, y es esencial para entender el razonamiento detrás de cada calificación, además de servir como texto de entrenamiento para modelos capaces de evaluar justificaciones.

El tratamiento de esta columna es similar a 'statement' y 'speaker_description' se transforma primero homogenizando el texto a minúsculas 
y suprimiendo signos de puntuación, paréntesis y símbolos que no aportan valor informativo, de modo que la justificación queda libre de ruido.

A continuación, aplicamos tokenización para segmentar cada justificación en sus unidades léxicas, preservando cifras y referencias como tokens relevantes.

Tras ello eliminamos las palabras vacías más frecuentes en inglés y lematizamos cada token con WordNet, unificando variantes y reduciendo la dispersión de términos.

Finalmente, reconstruimos la cadena limpia y generamos las variables que representan la longitud de caracteres de la cadena y la cantidad de 
palabras.

Este enfoque es el más adecuado porque permite al modelo centrarse en los términos clave de la evidencia sin distraerse con elementos tipográficos o muletillas, 
a la vez que agrupa formas flexivas para reducir el tamaño de la matriz documento-término. 

De este modo, maximizamos la señal de las justificaciones aportadas por los verificadores y garantizamos que el pipeline de TF-IDF o BoW extraiga pesos 
más precisos para distinguir motivo y alcance de cada verificación.