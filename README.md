# Tropical Weather Monitor — Datos meteorológicos

Generador automatizado de viento GFS (NOAA/NCEP) y geometrías/centros NHC.

## Instalación

1. Subir todos los archivos de este paquete al repositorio público `LuisCasas592/tropical-weather-monitor`, **conservando** la carpeta `.github/workflows/`.
2. En Settings → Pages, seleccionar GitHub Actions.
3. En Actions → «Actualizar meteorologia y publicar Pages» → Run workflow, ejecutar una vez.
4. Esperar a que el flujo finalice correctamente y verificar:
   https://luiscasas592.github.io/tropical-weather-monitor/data/meteorologia.json

El flujo se programa cada 6 horas (puede demorarse). Si falla la descarga NOAA y no hay caché, la ejecución falla y Pages conserva la última publicación correcta.

**Importante:** La información no sustituye avisos oficiales. Los vórtices GFS no son posiciones oficiales de ciclones. Las coordenadas NHC se muestran solo cuando la fuente entrega campos válidos. Se requiere comprobar en vivo la estructura de `CurrentStorms.json` y la cobertura de los KML; la ausencia de marcadores no prueba ausencia de ciclones.

La URL JSON debe configurarse en el notebook HTML como `URL_DATOS_PUBLICOS`. No publicar archivos GRIB ni datos personales.
