# ED.03.02 — RDD en Apache Spark
# Asignatura: Manejo Masivo de Datos
# Alumno: Betel Zurisadai Hernández Sánchez
# Ejecutar en Google Colab (primero: pip install pyspark)


# ======================================================================
# ED.03.02 — RDD en Apache Spark
# **Alumno/a:** _(tu nombre)_
# **Contenido:** 8 Ejercicios Prácticos · Proyecto Práctico: Análisis de Calificaciones · Pregunta de reflexión (RDD vs DataFrame)
# ======================================================================
from pyspark.sql import SparkSession
from pyspark import StorageLevel
import os, shutil

spark = SparkSession.builder.appName("ED.03.02-RDD").master("local[*]").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
print("Spark", spark.version)


# ======================================================================
# Ejercicio 1: Word Count Básico
# Cuenta cuántas veces aparece cada palabra.
# `flatMap` separa las frases en palabras → `map` crea pares `(palabra, 1)` → `reduceByKey` suma por palabra.
# ======================================================================
texto = ["hola mundo", "hola spark", "mundo spark"]
rdd_texto = sc.parallelize(texto)

conteo = (rdd_texto
          .flatMap(lambda frase: frase.split(" "))   # ['hola','mundo','hola','spark',...]
          .map(lambda palabra: (palabra, 1))         # ('hola',1), ('mundo',1), ...
          .reduceByKey(lambda a, b: a + b))          # suma por clave

for palabra, n in sorted(conteo.collect()):
    print(f"{palabra}: {n}")


# ======================================================================
# Ejercicio 2: Análisis de Ventas
# 1. Suma las ventas por ciudad. 2. Filtra ciudades con ventas > 100. 3. Ordena de mayor a menor.
# ======================================================================
ventas = sc.parallelize([("Madrid", 100), ("Barcelona", 200), ("Madrid", 50)])

# 1. Suma por ciudad
ventas_ciudad = ventas.reduceByKey(lambda a, b: a + b)
print("1. Suma por ciudad:", ventas_ciudad.collect())

# 2. Ciudades con ventas > 100
ventas_mas_100 = ventas_ciudad.filter(lambda x: x[1] > 100)
print("2. Ventas > 100:   ", ventas_mas_100.collect())

# 3. Orden de mayor a menor
ordenadas = ventas_mas_100.sortBy(lambda x: x[1], ascending=False)
print("3. Ordenadas:      ", ordenadas.collect())


# ======================================================================
# Ejercicio 3: Filtrado y Transformación
# Filtra los precios ≥ 20 USD y conviértelos a EUR (×0.9).
# ======================================================================
precios_usd = sc.parallelize([10, 25, 40, 5, 100, 15])

precios_eur = (precios_usd
               .filter(lambda p: p >= 20)            # [25, 40, 100]
               .map(lambda p: round(p * 0.9, 2)))    # a EUR

print("Precios USD >= 20:", precios_usd.filter(lambda p: p >= 20).collect())
print("Precios en EUR:   ", precios_eur.collect())


# ======================================================================
# Ejercicio 4: Procesamiento de CSV
# Divide, aplana y cuenta productos únicos.
# ======================================================================
compras = sc.parallelize(["Leche,Huevos,Pan", "Manzanas,Peras", "Leche,Chocolate"])

productos = compras.flatMap(lambda linea: linea.split(","))   # dividir + aplanar
print("Todos los productos:", productos.collect())

unicos = productos.distinct()
print("Productos únicos:   ", sorted(unicos.collect()))
print("Nº productos únicos:", unicos.count())

# Extra: cuántas veces aparece cada producto
print("Frecuencia:", sorted(productos.map(lambda p: (p, 1)).reduceByKey(lambda a, b: a + b).collect()))


# ======================================================================
# Ejercicio 5: Análisis de Logs
# Formato `ip,hora,metodo,url`. 1. Cargar el archivo · 2. Peticiones por IP (`map` + `reduceByKey`) · 3. IPs con más de 1 petición · 4. Guardar en un archivo de texto.
# ======================================================================
# Creamos el archivo simulado
with open("logs.txt", "w") as f:
    f.write("192.168.1.1,10:00,GET,/index\n")
    f.write("192.168.1.2,10:01,POST,/login\n")
    f.write("192.168.1.1,10:02,GET,/imagen.jpg\n")

# 1. Cargar el archivo
logs = sc.textFile("logs.txt")
print("1. Líneas cargadas:", logs.collect())

# 2. Peticiones por IP
peticiones_ip = (logs
                 .map(lambda linea: (linea.split(",")[0], 1))
                 .reduceByKey(lambda a, b: a + b))
print("2. Peticiones por IP:", peticiones_ip.collect())

# 3. IPs con más de 1 petición
ips_activas = peticiones_ip.filter(lambda x: x[1] > 1)
print("3. IPs con > 1 petición:", ips_activas.collect())

# 4. Guardar resultado (saveAsTextFile crea una carpeta; la borramos si ya existe)
salida = "resultado_logs"
if os.path.exists(salida):
    shutil.rmtree(salida)
ips_activas.map(lambda x: f"{x[0]},{x[1]}").coalesce(1).saveAsTextFile(salida)
print("4. Contenido guardado:", sc.textFile(salida).collect())


# ======================================================================
# Ejercicio 6: Pipeline y Shuffle
# 1. Filtra (descarta) las ventas menores de 60 · 2. Suma por ciudad · 3. Ordena por facturación (mayor a menor).
# Pipeline: `filter` (narrow, sin shuffle) → `reduceByKey` (wide, shuffle) → `sortBy` (wide, shuffle).
# Filtrar **antes** del `reduceByKey` reduce los datos que viajan por la red en el shuffle.
# ======================================================================
ventas = sc.parallelize([("Madrid", 100), ("Barcelona", 200), ("Madrid", 50), ("Valencia", 50)])

resultado = (ventas
             .filter(lambda x: x[1] >= 60)                    # 1. quitamos las ventas < 60
             .reduceByKey(lambda a, b: a + b)                 # 2. suma por ciudad
             .sortBy(lambda x: x[1], ascending=False))        # 3. mayor a menor

print(resultado.collect())
print()
print(resultado.toDebugString().decode())   # lineaje: se ven los stages separados por el shuffle


# ======================================================================
# Ejercicio 7: Diferencia Map vs FlatMap
# ======================================================================
frases = sc.parallelize(["Hola mundo", "Apache Spark es guay"])

con_map = frases.map(lambda f: f.split(" "))
print("map     ->", con_map.collect(), "| elementos:", con_map.count())

con_flatmap = frases.flatMap(lambda f: f.split(" "))
print("flatMap ->", con_flatmap.collect(), "| elementos:", con_flatmap.count())


# ======================================================================
# **Respuesta:**
# 1. Con `map` obtenemos una **lista de listas**: cada frase (1 elemento de entrada) se convierte en 1 elemento de salida, que es una lista de palabras → 2 elementos.
# 2. Con `flatMap` el resultado se **aplana**: cada frase genera N elementos y en el `collect()` vemos una lista plana de palabras sueltas → 6 elementos. Por eso para contar palabras se usa `flatMap`.
# ======================================================================

# ======================================================================
# Ejercicio 8: Optimización
# **Problema:** tengo un RDD que uso 5 veces. Por la *lazy evaluation*, cada acción recalcula todo el lineaje desde el origen (5 veces leer + transformar).
# **Solución:** persistirlo en memoria con `cache()` (o `persist(StorageLevel.MEMORY_AND_DISK)` si es grande y puede no caber en RAM). La primera acción lo calcula y lo guarda; las 4 siguientes lo leen ya calculado. Al terminar, liberar con `unpersist()`.
# ======================================================================
import time

base = sc.parallelize(range(1, 2_000_001)).map(lambda x: x * 2).filter(lambda x: x % 3 == 0)

base.persist(StorageLevel.MEMORY_AND_DISK)   # equivalente a cache() pero con disco de respaldo

t0 = time.time()
print("count:", base.count())                # 1ª acción: calcula y guarda en caché
print("sum:  ", base.sum())                  # siguientes: leen de caché
print("max:  ", base.max())
print("min:  ", base.min())
print("media:", base.mean())
print(f"Tiempo total 5 acciones: {time.time() - t0:.2f} s")
print("¿Está cacheado?", base.is_cached, "|", base.getStorageLevel())

base.unpersist()   # liberar memoria al terminar


# ======================================================================
# ---
# Proyecto Práctico: Análisis de Calificaciones
# **Escenario:** soy el administrador de un instituto con las notas de tres asignaturas en archivos de texto separados y *sucios* (formato `nombre,nota`). Hay que unificar los datos y generar un informe estadístico usando RDDs.
# > Nota: los archivos se generan aquí con datos de ejemplo e incluyen "suciedad" típica (espacios, mayúsculas/minúsculas, líneas vacías, notas no numéricas, coma decimal, notas fuera de rango) para que la limpieza tenga sentido. Si el profesor dio otras notas, basta con cambiar el contenido de los archivos en la siguiente celda.
# ======================================================================
archivos = {
    "notas_mates.txt":  "Angel,8\nMaria,9\nRamon,4.5\n  luis , 6\n\nPedro,abc\nAna,11\n",
    "notas_fisica.txt": "Angel,7\nMaria,10\nLUIS,3\nRamon,5,5\n\n",
    "notas_ingles.txt": "Angel,6\nMaria,8\nluis,7.5\n;;\nAna,9\n",
}
for nombre, contenido in archivos.items():
    with open(nombre, "w", encoding="utf-8") as f:
        f.write(contenido)
    print(f"--- {nombre} ---\n{contenido}")


# ======================================================================
# Paso 1 — Parseo y limpieza
# Función `parsear` que devuelve una tupla `(nombre, nota)` o `None` si la línea no es válida. Un **accumulator** cuenta las líneas descartadas.
# ======================================================================
lineas_erroneas = sc.accumulator(0)

def parsear(linea):
    linea = linea.strip()
    if not linea:
        return None                      # línea vacía (no la contamos como error)
    partes = linea.split(",", 1)         # solo el primer separador: "Ramon,5,5" -> ["Ramon", "5,5"]
    if len(partes) != 2:
        lineas_erroneas.add(1)
        return None
    nombre = partes[0].strip().capitalize()
    try:
        nota = float(partes[1].strip().replace(",", "."))   # coma decimal -> punto
    except ValueError:
        lineas_erroneas.add(1)
        return None
    if not nombre or not (0 <= nota <= 10):
        lineas_erroneas.add(1)
        return None
    return (nombre, nota)

def cargar(ruta):
    return sc.textFile(ruta).map(parsear).filter(lambda x: x is not None).cache()  # cache: el accumulator no se cuenta dos veces

mates  = cargar("notas_mates.txt")
fisica = cargar("notas_fisica.txt")
ingles = cargar("notas_ingles.txt")

print("Mates: ", mates.collect())
print("Física:", fisica.collect())
print("Inglés:", ingles.collect())
print("Líneas erróneas descartadas:", lineas_erroneas.value)


# ======================================================================
# Paso 2 — Unificar los datos
# Añadimos la asignatura y unimos los tres RDDs con `union` (narrow, no hay shuffle).
# ======================================================================
todas = (mates.map(lambda x: (x[0], ("Mates", x[1])))
         .union(fisica.map(lambda x: (x[0], ("Física", x[1]))))
         .union(ingles.map(lambda x: (x[0], ("Inglés", x[1])))))
todas.cache()   # lo vamos a usar muchas veces (Ejercicio 8)

for registro in todas.collect():
    print(registro)


# ======================================================================
# Paso 3 — Nota media por alumno
# Usamos `reduceByKey` sobre pares `(suma, contador)` (pre-agrega en cada partición, mejor que `groupByKey`).
# ======================================================================
media_alumno = (todas
                .mapValues(lambda v: (v[1], 1))
                .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
                .mapValues(lambda s: round(s[0] / s[1], 2))
                .sortBy(lambda x: x[1], ascending=False))

for nombre, media in media_alumno.collect():
    print(f"{nombre:<6} -> {media}")


# ======================================================================
# Paso 4 — Estadísticas por asignatura
# Media, máxima y mínima de cada asignatura con `aggregateByKey`.
# ======================================================================
por_asignatura = todas.map(lambda x: (x[1][0], x[1][1]))   # (asignatura, nota)

stats = por_asignatura.aggregateByKey(
    (0.0, 0, 10.0, 0.0),                                         # (suma, n, min, max)
    lambda acc, n: (acc[0] + n, acc[1] + 1, min(acc[2], n), max(acc[3], n)),
    lambda a, b: (a[0] + b[0], a[1] + b[1], min(a[2], b[2]), max(a[3], b[3])))

print(f"{'Asignatura':<10}{'Media':>7}{'Mín':>6}{'Máx':>6}{'Nº':>4}")
for asig, (suma, n, mn, mx) in sorted(stats.collect()):
    print(f"{asig:<10}{suma/n:>7.2f}{mn:>6}{mx:>6}{n:>4}")


# ======================================================================
# Paso 5 — Mejor alumno de cada asignatura
# ======================================================================
mejor = (todas
         .map(lambda x: (x[1][0], (x[0], x[1][1])))          # (asignatura, (alumno, nota))
         .reduceByKey(lambda a, b: a if a[1] >= b[1] else b))

for asig, (alumno, nota) in sorted(mejor.collect()):
    print(f"{asig}: {alumno} ({nota})")


# ======================================================================
# Paso 6 — Operaciones de conjuntos
# - Alumnos que aparecen en las tres asignaturas → `intersection`
# - Alumnos que no tienen nota en alguna asignatura → `subtract`
# ======================================================================
alumnos_mates  = mates.keys().distinct()
alumnos_fisica = fisica.keys().distinct()
alumnos_ingles = ingles.keys().distinct()

todos_los_alumnos = alumnos_mates.union(alumnos_fisica).union(alumnos_ingles).distinct()
en_las_tres = alumnos_mates.intersection(alumnos_fisica).intersection(alumnos_ingles)
incompletos = todos_los_alumnos.subtract(en_las_tres)

print("Todos los alumnos:      ", sorted(todos_los_alumnos.collect()))
print("Con nota en las tres:   ", sorted(en_las_tres.collect()))
print("Les falta alguna nota:  ", sorted(incompletos.collect()))
print("Sin nota de Física:     ", sorted(todos_los_alumnos.subtract(alumnos_fisica).collect()))


# ======================================================================
# Paso 7 — Aprobados y suspensos
# ======================================================================
resumen = (todas
           .map(lambda x: ((x[1][0], "Aprobado" if x[1][1] >= 5 else "Suspenso"), 1))
           .reduceByKey(lambda a, b: a + b))
for (asig, estado), n in sorted(resumen.collect()):
    print(f"{asig:<7} {estado:<9} {n}")

suspensos = todas.filter(lambda x: x[1][1] < 5).map(lambda x: f"{x[0]} - {x[1][0]} ({x[1][1]})")
print("\nSuspensos:", suspensos.collect())


# ======================================================================
# Paso 8 — Informe final
# ======================================================================
medias = dict(media_alumno.collect())
lineas = ["INFORME DE CALIFICACIONES", "=" * 30]
for nombre in sorted(medias):
    notas = sorted(todas.filter(lambda x, n=nombre: x[0] == n).map(lambda x: x[1]).collect())
    detalle = ", ".join(f"{a}: {n}" for a, n in notas)
    estado = "APROBADO" if medias[nombre] >= 5 else "SUSPENSO"
    lineas.append(f"{nombre:<6} | media {medias[nombre]:>5} | {estado:<8} | {detalle}")
lineas.append("=" * 30)
lineas.append(f"Mejor media: {media_alumno.first()[0]} ({media_alumno.first()[1]})")
lineas.append(f"Líneas sucias descartadas: {lineas_erroneas.value}")

informe = "\n".join(lineas)
print(informe)

salida = "informe_calificaciones"
if os.path.exists(salida):
    shutil.rmtree(salida)
sc.parallelize(lineas, 1).saveAsTextFile(salida)

todas.unpersist()


# ======================================================================
# ---
# Pregunta de reflexión: ¿cuándo usaría un RDD en lugar de un DataFrame?
# Usaría un RDD cuando los datos no tienen un esquema fijo y tengo que limpiarlos línea a línea con lógica propia de Python antes de poder darles estructura. Un ejemplo es el **análisis de logs** (Ejercicio 5) o los **archivos de notas sucios** del proyecto: hay líneas vacías, separadores que faltan, comas decimales y valores no numéricos, y con `map` + una función `parsear` controlo exactamente qué hago con cada línea. Un DataFrame necesita un esquema desde el principio y aquí fallaría o tendría que hacer muchos apaños. En cuanto los datos ya estén limpios y tabulares, pasaría a DataFrame para aprovechar el optimizador Catalyst.
# ======================================================================
spark.stop()

