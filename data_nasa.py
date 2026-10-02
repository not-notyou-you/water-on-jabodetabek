# data_nasa.py
import math
from datetime import date, timedelta
from urllib.parse import urlparse

import numpy as np
import requests

CMR_GRANULE = 'https://cmr.earthdata.nasa.gov/search/granules.json'
HOST_URS = 'urs.earthdata.nasa.gov'
PRODUK_GPM = (('GPM_3IMERGDF', 'Final'), ('GPM_3IMERGDL', 'Late'), ('GPM_3IMERGDE', 'Early'))
ORNL = 'https://modis.ornl.gov/rst/api/v1'
PRODUK_MODIS = 'MOD09A1'
R_SINUSOIDAL = 6371007.181
BATAS_WAKTU = 90
ISI_KOSONG_GPM = -9999.0
ISI_KOSONG_MODIS = -28672


class GagalUnduh(Exception):
    pass


class SesiEarthdata(requests.Session):
    def __init__(self, user=None, password=None, token=None):
        super().__init__()
        self.headers['User-Agent'] = 'water-on-jabodetabek/1.0'
        if token:
            self.headers['Authorization'] = f'Bearer {token.strip()}'
        elif user and password:
            self.auth = (user.strip(), password)

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        asal = urlparse(response.request.url).hostname
        tujuan = urlparse(prepared_request.url).hostname
        if 'Authorization' in headers and asal != tujuan and HOST_URS not in (asal, tujuan):
            del headers['Authorization']


def kredensial_lengkap(kred):
    kred = kred or {}
    return bool(kred.get('token')) or bool(kred.get('user') and kred.get('password'))


def buat_sesi(kred):
    if not kredensial_lengkap(kred):
        raise GagalUnduh('Kredensial NASA Earthdata belum diisi. Isi token, atau username dan password.')
    return SesiEarthdata(kred.get('user'), kred.get('password'), kred.get('token'))


def _get(sesi, url, **kwargs):
    try:
        r = sesi.get(url, timeout=BATAS_WAKTU, **kwargs)
    except requests.RequestException as e:
        raise GagalUnduh(f'Koneksi ke {urlparse(url).hostname} gagal: {e.__class__.__name__}') from e
    if r.status_code in (401, 403):
        raise GagalUnduh('Akses ditolak NASA Earthdata (401/403). Periksa token atau username dan password, '
                         'dan pastikan aplikasi "NASA GESDISC DATA ARCHIVE" sudah disetujui di profil Earthdata.')
    return r


def cari_granule_gpm(tgl, short_name):
    params = {'short_name': short_name, 'version': '07', 'page_size': 5,
              'temporal': f'{tgl.isoformat()}T00:00:00Z,{tgl.isoformat()}T23:59:59Z'}
    try:
        r = requests.get(CMR_GRANULE, params=params, timeout=BATAS_WAKTU)
        r.raise_for_status()
    except requests.RequestException as e:
        raise GagalUnduh(f'Pencarian granule GPM di CMR gagal: {e.__class__.__name__}') from e
    for g in r.json().get('feed', {}).get('entry', []):
        if tgl.strftime('%Y%m%d') not in g.get('title', ''):
            continue
        for l in g.get('links', []):
            href = l.get('href', '')
            if l.get('rel', '').endswith('service#') and 'opendap' in href and '/granules/' in href and not l.get('inherited'):
                return href
    return None


def indeks_imerg(extent):
    lon_min, lon_maks, lat_min, lat_maks = extent
    i0 = max(0, int(math.floor((lon_min + 179.95) / 0.1)) - 1)
    i1 = min(3599, int(math.ceil((lon_maks + 179.95) / 0.1)) + 1)
    j0 = max(0, int(math.floor((lat_min + 89.95) / 0.1)) - 1)
    j1 = min(1799, int(math.ceil((lat_maks + 89.95) / 0.1)) + 1)
    lon_c = -179.95 + 0.1 * np.arange(i0, i1 + 1)
    lat_c = -89.95 + 0.1 * np.arange(j0, j1 + 1)
    return (i0, i1, j0, j1), lon_c, lat_c


def _angka(teks):
    try:
        return float(teks)
    except ValueError:
        return None


def urai_ascii_opendap(teks, n_lon, n_lat):
    nilai = []
    for baris in teks.splitlines():
        bagian = [b.strip() for b in baris.split(',') if b.strip() != '']
        if not bagian:
            continue
        sisa = bagian[1:] if '[' in bagian[0] else bagian
        angka = [_angka(b) for b in sisa]
        if sisa and all(a is not None for a in angka):
            nilai.extend(angka)
    if len(nilai) < n_lon * n_lat:
        raise GagalUnduh(f'Respons OPeNDAP tidak berisi {n_lon * n_lat} nilai (terbaca {len(nilai)}).')
    arr = np.asarray(nilai[-n_lon * n_lat:], np.float32).reshape(n_lon, n_lat)
    arr[(arr <= ISI_KOSONG_GPM + 1) | (arr < 0)] = np.nan
    return arr


def unduh_imerg_hari(sesi, tgl, extent):
    (i0, i1, j0, j1), lon_c, lat_c = indeks_imerg(extent)
    n_lon, n_lat = i1 - i0 + 1, j1 - j0 + 1
    kesalahan = []
    for short_name, label in PRODUK_GPM:
        url = cari_granule_gpm(tgl, short_name)
        if url is None:
            continue
        percobaan = [f'{url}.dap.csv?dap4.ce=/precipitation[0][{i0}:{i1}][{j0}:{j1}]',
                     f'{url}.ascii?precipitation[0:0][{i0}:{i1}][{j0}:{j1}]']
        for alamat in percobaan:
            r = _get(sesi, alamat)
            if r.status_code != 200:
                kesalahan.append(f'{label} HTTP {r.status_code}')
                continue
            try:
                arr = urai_ascii_opendap(r.text, n_lon, n_lat)
            except GagalUnduh as e:
                kesalahan.append(f'{label}: {e}')
                continue
            return {'lon': lon_c, 'lat': lat_c, 'nilai': arr, 'produk': label}
    if kesalahan:
        raise GagalUnduh(f'Data GPM {tgl.isoformat()} gagal diambil: ' + '; '.join(kesalahan))
    return None


def resample_bilinear(lon_c, lat_c, nilai, lon_px, lat_px):
    v = np.nan_to_num(nilai, nan=0.0)
    m = np.isfinite(nilai).astype(np.float32)
    v_lat = np.stack([np.interp(lat_px, lat_c, v[i]) for i in range(v.shape[0])])
    m_lat = np.stack([np.interp(lat_px, lat_c, m[i]) for i in range(m.shape[0])])
    v_out = np.stack([np.interp(lon_px, lon_c, v_lat[:, r]) for r in range(len(lat_px))])
    m_out = np.stack([np.interp(lon_px, lon_c, m_lat[:, r]) for r in range(len(lat_px))])
    return np.where(m_out > 0.999, v_out / np.maximum(m_out, 1e-6), np.nan).astype(np.float32)


def jendela_hujan(harian, tgl_akhir, lon_px, lat_px):
    hari = [tgl_akhir - timedelta(days=i) for i in range(7)]
    if any(harian.get(d) is None for d in hari):
        return None
    contoh = harian[hari[0]]
    nilai = [harian[d]['nilai'] for d in hari]
    jumlah = {'rf24': nilai[0], 'rf72': np.sum(nilai[:3], axis=0), 'rf7d': np.sum(nilai, axis=0)}
    peta = {x: resample_bilinear(contoh['lon'], contoh['lat'], v, lon_px, lat_px) for x, v in jumlah.items()}
    peta['produk'] = sorted({harian[d]['produk'] for d in hari})
    return peta


def _tanggal_modis(kode):
    return date(int(kode[1:5]), 1, 1) + timedelta(days=int(kode[5:8]) - 1)


def kandidat_komposit(tgl, jumlah=8):
    hasil = []
    tahun = tgl.year
    while len(hasil) < jumlah:
        for doy in range(361, 0, -8):
            d = date(tahun, 1, 1) + timedelta(days=doy - 1)
            if d.year == tahun and d <= tgl:
                hasil.append(f'A{tahun}{doy:03d}')
                if len(hasil) >= jumlah:
                    break
        tahun -= 1
    return hasil


def _subset_ornl(lat, lon, band, kode, km_ab, km_lr):
    params = {'latitude': f'{lat:.5f}', 'longitude': f'{lon:.5f}', 'band': band, 'startDate': kode,
              'endDate': kode, 'kmAboveBelow': km_ab, 'kmLeftRight': km_lr}
    try:
        r = requests.get(f'{ORNL}/{PRODUK_MODIS}/subset', params=params, headers={'Accept': 'application/json'},
                         timeout=BATAS_WAKTU * 2)
    except requests.RequestException as e:
        raise GagalUnduh(f'Koneksi ke layanan MODIS ORNL gagal: {e.__class__.__name__}') from e
    if r.status_code != 200:
        return None
    js = r.json()
    if not js.get('subset') or not js['subset'][0].get('data'):
        return None
    return js


def sampel_sinusoidal(js, lon_px, lat_px):
    nrows, ncols = int(js['nrows']), int(js['ncols'])
    xll, yll, cs = float(js['xllcorner']), float(js['yllcorner']), float(js['cellsize'])
    data = np.asarray(js['subset'][0]['data'], np.float64).reshape(nrows, ncols)
    lon2, lat2 = np.meshgrid(np.radians(lon_px), np.radians(lat_px))
    x = R_SINUSOIDAL * lon2 * np.cos(lat2)
    y = R_SINUSOIDAL * lat2
    kol = np.floor((x - xll) / cs).astype(np.int64)
    bar = nrows - 1 - np.floor((y - yll) / cs).astype(np.int64)
    ok = (kol >= 0) & (kol < ncols) & (bar >= 0) & (bar < nrows)
    out = np.full(lon2.shape, np.nan)
    out[ok] = data[bar[ok], kol[ok]]
    return out


def ambil_modis(tgl, extent, lon_px, lat_px, lapor=None):
    lon_min, lon_maks, lat_min, lat_maks = extent
    lat_c, lon_c = (lat_min + lat_maks) / 2, (lon_min + lon_maks) / 2
    km_ab = min(100, int(math.ceil((lat_maks - lat_min) / 2 * 110.57)) + 2)
    km_lr = min(100, int(math.ceil((lon_maks - lon_min) / 2 * 111.32 * math.cos(math.radians(lat_c)))) + 2)
    for kode in kandidat_komposit(tgl):
        if lapor:
            lapor(f'Mencari komposit MODIS {PRODUK_MODIS} {_tanggal_modis(kode).isoformat()}')
        js1 = _subset_ornl(lat_c, lon_c, 'sur_refl_b01', kode, km_ab, km_lr)
        if js1 is None:
            continue
        band = {'b01': sampel_sinusoidal(js1, lon_px, lat_px)}
        for nama, kode_band in (('b02', 'sur_refl_b02'), ('b04', 'sur_refl_b04'), ('state', 'sur_refl_state_500m')):
            if lapor:
                lapor(f'Mengunduh MODIS {kode_band}')
            js = _subset_ornl(lat_c, lon_c, kode_band, kode, km_ab, km_lr)
            if js is None:
                raise GagalUnduh(f'Band {kode_band} MODIS {kode} gagal diambil.')
            band[nama] = sampel_sinusoidal(js, lon_px, lat_px)
        refl = {}
        for k in ('b01', 'b02', 'b04'):
            a = band[k]
            a = np.where((a == ISI_KOSONG_MODIS) | (a < -100) | (a > 16000), np.nan, a * 1e-4)
            refl[k] = a
        state = band['state']
        awan = np.isnan(state) | (state == 65535) | ((np.nan_to_num(state, nan=0).astype(np.int64) & 0b111) != 0)
        for k in refl:
            refl[k][awan] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (refl['b02'] - refl['b01']) / (refl['b02'] + refl['b01'])
            ndwi = (refl['b04'] - refl['b02']) / (refl['b04'] + refl['b02'])
        ndvi = np.where(np.isfinite(ndvi), np.clip(ndvi, -1, 1), np.nan).astype(np.float32)
        ndwi = np.where(np.isfinite(ndwi), np.clip(ndwi, -1, 1), np.nan).astype(np.float32)
        return {'ndvi': ndvi, 'ndwi': ndwi, 'komposit': _tanggal_modis(kode), 'kode': kode}
    raise GagalUnduh('Komposit MODIS untuk tanggal tersebut belum tersedia.')