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

def obtener_nhc():
    ses=requests.Session();ses.headers.update({'User-Agent':'TropicalWeatherMonitor/0.4 educational'})
    try:
        r=ses.get(NHC_FEED,timeout=25);r.raise_for_status()
        root=_xml_kmz(r.content)
        links=[]
        for node in root.iter():
            if node.tag.split('}')[-1]!='NetworkLink':continue
            name=next((e.text for e in node if e.tag.split('}')[-1]=='name'),'') or ''
            href=next((e.text for e in node.iter() if e.tag.split('}')[-1]=='href'),None)
            if href:links.append((name,urljoin(NHC_FEED,href)))
        # Resolver enlaces anidados y reunir candidatos antes de dibujar.
        # No mezclar indiscriminadamente geometrías de avisos diferentes.
        queue=[(name,url,0) for name,url in links]; visited=set(); candidates={}
        while queue and len(visited)<65:
            name,url,depth=queue.pop(0)
            if url in visited:continue
            visited.add(url)
            lower=(name+' '+url).lower()
            if any(x in lower for x in ['warning','surge','probabilit','radii','swath','outlook','arrival']):continue
            try:
                rr=ses.get(url,timeout=20);rr.raise_for_status()
                sub=_xml_kmz(rr.content)
            except Exception:continue
            if depth<2:
                for node in sub.iter():
                    if node.tag.split('}')[-1]=='NetworkLink':
                        nm=next((e.text for e in node if e.tag.split('}')[-1]=='name'),'') or ''
                        href=next((e.text for e in node.iter() if e.tag.split('}')[-1]=='href'),None)
                        if href:queue.append((nm,urljoin(url,href),depth+1))
            kind='cone' if 'cone' in lower else 'track' if any(x in lower for x in ['track','forecast','fcst']) else 'other'
            if kind=='other':continue
            m=re.search(r'(AL|EP|CP)\d{2}20\d{2}',url,re.I)
            if not m:continue  # No atribuir geometría sin identificador inequívoco.
            sid=m.group(0).upper()
            geom=_placemarks(sub)
            lines=[pm['coords'] for pm in geom if pm['kind']=='LineString']
            polygons=[pm['coords'] for pm in geom if pm['kind']=='Polygon']
            pts=[{'coord':pm['coords'][0],'name':pm['name'],'description':pm['description']} for pm in geom if pm['kind']=='Point']
            if kind=='track' and not lines:continue
            if kind=='cone' and not polygons:continue
            candidates.setdefault(sid,{}).setdefault(kind,[]).append({'url':url,'lines':lines,'polygons':polygons,'points':pts})
        storms=[]
        for sid, kinds in candidates.items():
            # Seleccionar un solo archivo por tipo. Si hay varios avisos,
            # no combinar trayectorias: escoger el de nombre más reciente.
            def latest(items):
                return sorted(items,key=lambda x:x['url'])[-1] if items else None
            tr=latest(kinds.get('track',[])); co=latest(kinds.get('cone',[]))
            if not tr and not co:continue
            lines=(tr or {}).get('lines',[])
            polys=(co or {}).get('polygons',[])
            # Las posiciones de los KML pueden contener tiempos históricos.
            # No dibujar todos los puntos ni presentarlos como posiciones actuales.
            # Solo un punto ilustrativo al inicio de la línea, sin categoría inferida.
            storms.append({'id':sid,'name':sid,'track':lines[:1], 'cone':polys[:1],
                           'points':[], 'source_track':tr['url'] if tr else '',
                           'source_cone':co['url'] if co else ''})
        out={'status':'ok','consultado_utc':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), 'storms':storms}
        NHC_CACHE.write_text(json.dumps(out,ensure_ascii=False),encoding='utf-8')
        print('NHC: ciclones con geometría:',len(out['storms']))
        return out
    except Exception as exc:
        print('AVISO: NHC no disponible:',exc)
        if NHC_CACHE.exists():
            old=json.loads(NHC_CACHE.read_text(encoding='utf-8'));old['status']='cache';return old
        return {'status':'unavailable','consultado_utc':'','storms':[]}

def posiciones_oficiales_nhc():
    """Obtiene centros del listado de ciclones activos; nunca infiere centros del viento."""
    url = 'https://www.nhc.noaa.gov/CurrentStorms.json'
    try:
        r=requests.get(url,timeout=25,headers={'User-Agent':'TropicalWeatherMonitor/0.4 educational'})
        r.raise_for_status()
        raw=r.json()
        storms=raw.get('activeStorms',raw.get('currentStorms',[])) if isinstance(raw,dict) else []
        result=[]
        for s in storms:
            if not isinstance(s,dict):continue
            def first(*keys):
                return next((s[k] for k in keys if s.get(k) is not None),None)
            lat=first('latitude','lat');lon=first('longitude','lon','long')
            try: lat=float(lat);lon=float(lon)
            except (ValueError,TypeError):continue
            if not (-125<=lon<=-75 and 5<=lat<=34):continue
            result.append({'id':str(first('id','stormId','stormID') or ''),'name':str(first('name','stormName') or 'Ciclón'),
                'lat':lat,'lon':lon,'classification':first('classification','stormType','type'),
                'wind':first('intensity','maxWind','windSpeed'),'pressure':first('pressure','minimumPressure'),
                'updated':first('lastUpdate','advisoryDate','date')})
        return {'status':'ok','storms':result,'consultado_utc':datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        print('Aviso: posiciones oficiales NHC no disponibles:',str(exc)[:150])
        return {'status':'unavailable','storms':[]}
if __name__ == '__main__':
    wind = obtener_gfs()
    nhc = obtener_nhc()
    centers = posiciones_oficiales_nhc()
    payload = {'wind': wind, 'nhc': nhc, 'centers': centers,
               'published_utc': datetime.now(timezone.utc).isoformat(), 'schema_version': 1}
    path = OUTPUT / 'meteorologia.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print('Publicado:', path, 'bytes:', path.stat().st_size)
