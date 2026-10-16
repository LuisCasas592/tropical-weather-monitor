# Tropical Weather Monitor — generador remoto de datos
import json, math, re, io, zipfile, time
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode, urljoin
import requests
import numpy as np
import xarray as xr
BASE = Path(__file__).resolve().parent
INPUT = BASE / 'Input'
OUTPUT = BASE / 'data'
INPUT.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)
CACHE_JSON = INPUT / 'gfs_viento_mexico_cache.json'
SUR, NORTE, OESTE, ESTE = 5.0, 34.0, -125.0, -75.0
MAX_HORAS_CACHE = 9
NHC_FEED = 'https://www.nhc.noaa.gov/gis/kml/nhc_active.kml'
NHC_CACHE = INPUT / 'nhc_ciclones_activos_cache_v03.json'
def obtener_gfs():
    ahora = datetime.now(timezone.utc)
    sesion = requests.Session()
    sesion.headers.update({'User-Agent':'TropicalWeatherMonitor/0.4 (educational project)'})
    errores = []
    # Evitar ciclos todavía no publicados. Probar de más reciente a más antiguo.
    for desfase in [6,12,18,24,30,36,42]:
        instante = ahora - timedelta(hours=desfase)
        ciclo = (instante.hour // 6) * 6
        fecha = instante.strftime('%Y%m%d')
        hh = f'{ciclo:02d}'
        nombre = f'gfs.t{hh}z.pgrb2.0p25.f000'
        destino = INPUT / f'gfs_{fecha}_{hh}_mexico_10m.grib2'
        params = {
            'file': nombre, 'lev_10_m_above_ground':'on',
            'var_UGRD':'on', 'var_VGRD':'on',
            'subregion':'', 'leftlon':str(OESTE+360),
            'rightlon':str(ESTE+360), 'toplat':str(NORTE),
            'bottomlat':str(SUR), 'dir':f'/gfs.{fecha}/{hh}/atmos'
        }
        url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?' + urlencode(params)
        try:
            if not destino.exists() or destino.stat().st_size < 10000:
                print(f'Probando GFS {fecha} {hh}Z ...')
                with sesion.get(url, timeout=(15,90), stream=True) as r:
                    r.raise_for_status()
                    with open(destino.with_suffix('.tmp'),'wb') as f:
                        for trozo in r.iter_content(131072):
                            if trozo: f.write(trozo)
                temporal = destino.with_suffix('.tmp')
                with open(temporal,'rb') as f: cabecera=f.read(4)
                if cabecera != b'GRIB':
                    temporal.unlink(missing_ok=True)
                    raise ValueError('Respuesta no es un GRIB válido')
                temporal.replace(destino)
            # indexpath vacío evita generar índices auxiliares
            ds = xr.open_dataset(destino, engine='cfgrib', backend_kwargs={'indexpath':'', 'filter_by_keys':{'typeOfLevel':'heightAboveGround','level':10}})
            if 'u10' not in ds or 'v10' not in ds:
                raise ValueError('No se encontraron ambas componentes u10 y v10')
            lats = np.asarray(ds.latitude.values, dtype=float)
            lons = np.asarray(ds.longitude.values, dtype=float)
            u = np.asarray(ds['u10'].values, dtype=float).squeeze()
            v = np.asarray(ds['v10'].values, dtype=float).squeeze()
            ds.close()
            if lats.ndim != 1 or lons.ndim != 1 or u.ndim != 2 or v.shape != u.shape:
                raise ValueError('Cuadrícula inesperada')
            lons = np.where(lons > 180, lons-360, lons)
            if lats[0] > lats[-1]:
                lats=lats[::-1]; u=u[::-1,:]; v=v[::-1,:]
            if lons[0] > lons[-1]:
                orden=np.argsort(lons); lons=lons[orden]; u=u[:,orden]; v=v[:,orden]
            # 0.25 grados -> muestreo 0.5 grados para HTML más ligero
            paso = 2 if len(lats)>80 else 1
            lats=lats[::paso]; lons=lons[::paso]; u=u[::paso,::paso]; v=v[::paso,::paso]
            if not np.isfinite(u).all() or not np.isfinite(v).all():
                raise ValueError('Datos de viento incompletos')
            datos = {
                'fuente':'NOAA/NCEP GFS 0.25° (reducido para visualización)',
                'modelo_utc':f'{fecha[:4]}-{fecha[4:6]}-{fecha[6:]} {hh}:00 UTC',
                'descargado_utc':ahora.strftime('%Y-%m-%d %H:%M UTC'),
                'lat0':float(lats[0]),'lon0':float(lons[0]),
                'dlat':float(lats[1]-lats[0]),'dlon':float(lons[1]-lons[0]),
                'ny':len(lats),'nx':len(lons),
                'u':np.round(u,2).tolist(),'v':np.round(v,2).tolist()
            }
            CACHE_JSON.write_text(json.dumps(datos,separators=(',',':')),encoding='utf-8')
            print('GFS descargado y procesado correctamente:', datos['modelo_utc'])
            return datos
        except Exception as exc:
            errores.append(f'{fecha} {hh}Z: {str(exc)[:110]}')
    if CACHE_JSON.exists():
        cache=json.loads(CACHE_JSON.read_text(encoding='utf-8'))
        print('AVISO: NOAA no disponible. Se utilizará el último campo almacenado:',cache['modelo_utc'])
        return cache
    raise RuntimeError('No fue posible obtener GFS ni existe caché. Verifica internet y dependencias.\n'+'\n'.join(errores[-4:]))

def _xml_kmz(response_bytes):
    if response_bytes[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(response_bytes)) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.kml')]
            if not names: raise ValueError('KMZ sin KML')
            return ET.fromstring(z.read(names[0]))
    return ET.fromstring(response_bytes)

def _coords(s):
    out=[]
    for token in (s or '').strip().split():
        try:
            a=token.split(',');out.append([float(a[1]),float(a[0])])
        except (ValueError,IndexError):pass
    return out

def _placemarks(root):
    results=[]
    for pm in root.iter():
        if pm.tag.split('}')[-1]!='Placemark':continue
        name=next((e.text for e in pm if e.tag.split('}')[-1]=='name'),'') or ''
        desc=next((e.text for e in pm if e.tag.split('}')[-1]=='description'),'') or ''
        for node in pm.iter():
            kind=node.tag.split('}')[-1]
            if kind not in ('Point','LineString','Polygon'):continue
            coord=next((e.text for e in node.iter() if e.tag.split('}')[-1]=='coordinates'),None)
            points=_coords(coord)
            if points:results.append({'name':name,'description':re.sub('<[^>]*>',' ',desc)[:450], 'kind':kind,'coords':points})
    return results

def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _nhc_position(storm):
    """Usar los campos numéricos publicados explícitamente por el NHC."""
    lat=_number(storm.get('latitudeNumeric'))
    lon=_number(storm.get('longitudeNumeric'))
    if lat is None or lon is None:
        def parse_cardinal(s, positive, negative):
            if not isinstance(s,str): return _number(s)
            m=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(['+positive+negative+r'])\s*',s,re.I)
            if not m:return None
            return float(m.group(1))*(1 if m.group(2).upper()==positive else -1)
        lat=parse_cardinal(storm.get('latitude'),'N','S')
        lon=parse_cardinal(storm.get('longitude'),'E','W')
    if lat is None or lon is None or not (-90<=lat<=90 and -180<=lon<=180):return None
    return lat,lon

def _download_kmz(ses,url):
    if not url or not url.startswith('https://www.nhc.noaa.gov/'):return []
    r=ses.get(url,timeout=(15,35));r.raise_for_status()
    if len(r.content)>12_000_000:raise ValueError('KMZ demasiado grande')
    return _placemarks(_xml_kmz(r.content))

def obtener_productos_nhc():
    """Fuente única NHC: cada tormenta enlaza su propio KMZ de trayectoria y cono.
    Nunca se asignan centros basados en un vértice ni en el viento GFS.
    """
    now=datetime.now(timezone.utc).isoformat()
    ses=requests.Session()
    ses.headers.update({'User-Agent':'TropicalWeatherMonitor/0.5 (meteorological visualization)'})
    try:
        r=ses.get('https://www.nhc.noaa.gov/CurrentStorms.json',timeout=(15,35))
        r.raise_for_status();raw=r.json()
        if not isinstance(raw,dict) or not isinstance(raw.get('activeStorms'),list):
            raise ValueError('Estructura inesperada de CurrentStorms.json')
    except Exception as exc:
        print('NHC no disponible:',exc)
        return ({'status':'unavailable','consultado_utc':now,'storms':[]},
                {'status':'unavailable','consultado_utc':now,'storms':[]})
    centers=[];products=[]
    for storm in raw['activeStorms']:
        if not isinstance(storm,dict):continue
        pos=_nhc_position(storm)
        if not pos:continue
        lat,lon=pos
        if not (SUR<=lat<=NORTE and OESTE<=lon<=ESTE):continue
        sid=str(storm.get('id') or '').upper()
        if not re.fullmatch(r'(AL|EP|CP)\d{2}20\d{2}',sid):continue
        def url(section):
            item=storm.get(section) or {}
            return item.get('kmzFile') if isinstance(item,dict) else None
        center={'id':sid,'name':storm.get('name') or sid,'lat':lat,'lon':lon,
                'classification':storm.get('classification'),
                'wind':_number(storm.get('intensity')),
                'pressure':_number(storm.get('pressure')),
                'updated':storm.get('lastUpdate'),
                'advisory_url':(storm.get('publicAdvisory') or {}).get('url'),
                'forecast_url':(storm.get('forecastGraphics') or {}).get('url')}
        centers.append(center)
        track=[];cone=[];points=[];errors=[]
        for section,kind in [('forecastTrack','track'),('trackCone','cone')]:
            link=url(section)
            if not link:continue
            try:
                marks=_download_kmz(ses,link)
                if kind=='track':
                    track=[pm['coords'] for pm in marks if pm['kind']=='LineString' and len(pm['coords'])>=2][:5]
                    points=[{'coord':pm['coords'][0],'name':pm['name'],
                             'description':pm['description']} for pm in marks
                            if pm['kind']=='Point' and len(pm['coords'])==1][:35]
                else:
                    cone=[pm['coords'] for pm in marks if pm['kind']=='Polygon' and len(pm['coords'])>=3][:5]
            except Exception as exc:
                errors.append(kind+': '+str(exc)[:90])
        products.append({'id':sid,'name':center['name'],'track':track,'cone':cone,
                         'points':points,'source_track':url('forecastTrack') or '',
                         'source_cone':url('trackCone') or '',
                         'advisory_utc':(storm.get('forecastTrack') or {}).get('issuance'),
                         'errors':errors})
        print('NHC',sid,center['name'],'centro',lat,lon,
              'trayectorias',len(track),'cono',len(cone),'puntos',len(points),
              'avisos',errors)
    return ({'status':'ok','consultado_utc':now,'storms':products},
            {'status':'ok','consultado_utc':now,'storms':centers})

if __name__ == '__main__':
    wind = obtener_gfs()
    nhc, centers = obtener_productos_nhc()
    payload = {'wind': wind, 'nhc': nhc, 'centers': centers,
               'published_utc': datetime.now(timezone.utc).isoformat(), 'schema_version': 2}
    path = OUTPUT / 'meteorologia.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print('Publicado:', path, 'bytes:', path.stat().st_size)
