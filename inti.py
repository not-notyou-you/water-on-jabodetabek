# inti.py
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

KANAL_HUJAN = ('rf24', 'rf72', 'rf7d')
KANAL_MODIS = ('ndvi', 'ndwi', 'ndwi_valid')
NAMA_BULAN = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus',
              'September', 'Oktober', 'November', 'Desember']
NAMA_BULAN_PENDEK = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
KELAS_GENANGAN = ['Kering', 'Tergenang', 'Laut']
KELAS_ANOMALI = ['Kering seperti biasa', 'Tergenang seperti biasa', 'Lebih basah dari biasa',
                 'Lebih kering dari biasa', 'Laut']
KELAS_SELISIH = ['Tetap kering', 'Genangan baru', 'Tetap tergenang', 'Surut', 'Laut']
KATEGORI = ['Di bawah normal', 'Normal', 'Di atas normal']
JUMLAH_THREAD = 2


def _muat_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def muat_artefak(folder):
    folder = Path(folder)
    meta = _muat_json(folder / 'meta.json')
    with np.load(folder / 'kondisi_awal.npz') as d:
        kondisi = {k: d[k] for k in d.files}
    ch = meta['frame_ch']
    H, W = int(meta['grid']['H']), int(meta['grid']['W'])
    riwayat = np.zeros((meta['seq_len'] - 1, len(ch), H, W), np.float16)
    for t in range(meta['seq_len'] - 1):
        files = sorted(folder.glob(f'frame_{t}_*.npz'))
        if not files:
            raise FileNotFoundError(f'File frame_{t}_*.npz tidak ditemukan di {folder}')
        terisi = set()
        for p in files:
            with np.load(p) as d:
                for k in d.files:
                    if k not in ch or k in terisi:
                        raise ValueError(f'Kanal {k} pada {p.name} tidak valid atau ganda')
                    riwayat[t, ch.index(k)] = d[k]
                    terisi.add(k)
        if terisi != set(ch):
            raise ValueError(f'Kanal frame {t} tidak lengkap: {sorted(set(ch) - terisi)}')
    with np.load(folder / 'normal.npz') as d:
        normal = {k: d[k] for k in d.files}
    akurasi = {}
    if (folder / 'akurasi.npz').exists():
        with np.load(folder / 'akurasi.npz') as d:
            akurasi = {k: d[k] for k in d.files}
    model = torch.jit.load(str(folder / 'model.ts'), map_location='cpu').eval()
    return {'meta': meta, 'kondisi': kondisi, 'riwayat': riwayat, 'normal': normal, 'akurasi': akurasi, 'model': model}


def muat_verify(folder):
    return _muat_json(Path(folder) / 'verify.json')


def _norm(meta, kanal):
    if kanal in meta.get('norm', {}):
        return meta['norm'][kanal]
    if kanal in meta.get('norm_modis', {}):
        return meta['norm_modis'][kanal]
    if kanal == 'hist_freq':
        return meta['norm_hist']
    raise KeyError(kanal)


def z_mentah(meta, kanal, nilai):
    m, s = _norm(meta, kanal)
    a = np.asarray(nilai).astype(np.float16).astype(np.float32)
    return np.nan_to_num((a - np.float32(m)) / np.float32(s), nan=0.0)


def inferensi_bertile(model, riwayat, target, ukuran_tile, overlap_tile, lapor=None):
    torch.set_num_threads(JUMLAH_THREAD)
    T = riwayat.shape[0] + 1
    C, H, W = target.shape
    akumulasi = np.zeros((H, W), np.float32)
    hitung = np.zeros((H, W), np.float32)
    stride = ukuran_tile - 2 * overlap_tile
    x = np.zeros((T, C, ukuran_tile, ukuran_tile), np.float32)
    posisi = [(r, c) for r in range(0, H, stride) for c in range(0, W, stride)]
    with torch.inference_mode():
        for i, (r, c) in enumerate(posisi):
            r0, c0 = max(r - overlap_tile, 0), max(c - overlap_tile, 0)
            r1, c1 = min(r0 + ukuran_tile, H), min(c0 + ukuran_tile, W)
            h, w = r1 - r0, c1 - c0
            x.fill(0)
            x[:T - 1, :, :h, :w] = riwayat[:, :, r0:r1, c0:c1]
            x[T - 1, :, :h, :w] = target[:, r0:r1, c0:c1]
            p = torch.sigmoid(model(torch.from_numpy(x)[None]))[0].numpy()
            akumulasi[r0:r1, c0:c1] += p[:h, :w]
            hitung[r0:r1, c0:c1] += 1
            if lapor is not None:
                lapor((i + 1) / len(posisi))
    return akumulasi / np.maximum(hitung, 1)


def frame_target_verify(artefak, skenario):
    meta, ka = artefak['meta'], artefak['kondisi']
    ch = meta['frame_ch']
    darat = ka['darat']
    fr = np.zeros((len(ch), *darat.shape), np.float32)
    for x in KANAL_HUJAN:
        m, s = meta['norm'][x]
        fr[ch.index(x)] = (ka[f'pola_{x}'].astype(np.float32) * float(skenario[x]) - m) / s
    for kanal, kunci in (('ndvi', 'ndvi_target'), ('ndwi', 'ndwi_target'), ('ndwi_valid', 'ndwi_valid_target'),
                         ('land_dist', 'land_dist_target'), ('hist_freq', 'hist_target')):
        fr[ch.index(kanal)] = ka[kunci].astype(np.float32)
    m, s = meta['norm']['gap_days']
    fr[ch.index('gap_days')] = (float(skenario['gap_days']) - m) / s
    fr[:, ~darat] = 0
    return fr.astype(np.float16)


def cek_verifikasi(artefak, verify):
    meta = artefak['meta']
    prob = inferensi_bertile(artefak['model'], artefak['riwayat'], frame_target_verify(artefak, verify['skenario']),
                             int(meta['ukuran_tile']), int(meta['overlap_tile']))
    darat = artefak['kondisi']['darat']
    air = (prob >= meta['threshold']) & darat
    persen = air.sum() / max(darat.sum(), 1) * 100
    idx = np.asarray(verify['indeks_sampel'], np.int64)
    sel_persen = abs(float(persen) - verify['persen_tergenang'])
    sel_prob = float(np.max(np.abs(prob.ravel()[idx] - np.asarray(verify['prob_sampel'], np.float32))))
    return {'lulus': sel_persen <= verify['toleransi_persen'] and sel_prob <= verify['toleransi_prob'],
            'selisih_persen': sel_persen, 'selisih_prob_maks': sel_prob}


def ke_tanggal(teks):
    if isinstance(teks, datetime):
        return teks.date()
    if isinstance(teks, date):
        return teks
    return datetime.strptime(str(teks), '%Y%m%d').date()


def tanggal_riwayat_akhir(meta):
    return ke_tanggal(meta['tanggal_riwayat'][-1])


def format_tanggal(tgl, pendek=False):
    tgl = ke_tanggal(tgl)
    if pendek:
        return f'{tgl.day} {NAMA_BULAN_PENDEK[tgl.month - 1]} {tgl.year}'
    return f'{tgl.day} {NAMA_BULAN[tgl.month - 1]} {tgl.year}'


def periksa_tanggal(meta, daftar):
    awal = tanggal_riwayat_akhir(meta)
    gmin = int(meta['rantai']['gap_min'])
    urut = sorted(set(ke_tanggal(d) for d in daftar))
    masalah = []
    sebelumnya = awal
    for d in urut:
        if (d - sebelumnya).days < gmin:
            if sebelumnya == awal:
                masalah.append(f'{format_tanggal(d)} terlalu dekat dengan observasi terakhir '
                               f'({format_tanggal(awal)}); minimal {gmin} hari setelahnya.')
            else:
                masalah.append(f'{format_tanggal(d)} dan {format_tanggal(sebelumnya)} berjarak kurang dari {gmin} hari.')
        sebelumnya = d
    return urut, masalah


def rencana_langkah(meta, tanggal_urut):
    langkah = int(meta['rantai']['langkah_hari'])
    gmin, gmaks = int(meta['rantai']['gap_min']), int(meta['rantai']['gap_maks'])
    rencana = []
    sebelumnya = tanggal_riwayat_akhir(meta)
    ke = 0
    for d in tanggal_urut:
        total = (d - sebelumnya).days
        n = max(1, math.ceil(total / langkah))
        while total / n < gmin and n > 1:
            n -= 1
        while total / n > gmaks:
            n += 1
        dasar, sisa = divmod(total, n)
        gaps = [dasar + (1 if i < sisa else 0) for i in range(n)]
        t = sebelumnya
        for i, g in enumerate(gaps):
            t = t + timedelta(days=g)
            ke += 1
            rencana.append({'tanggal': t, 'gap': g, 'langkah_ke': ke, 'target': i == n - 1})
        sebelumnya = d
    return rencana


def keadaan_awal(artefak):
    normal = artefak['normal']
    n = normal['n_obs'].astype(np.float32)
    h = np.nan_to_num(normal['hist_mentah'].astype(np.float32), nan=0.0)
    return {'riwayat': artefak['riwayat'].copy(), 'cum_w': np.round(h * n).astype(np.float32), 'cum_n': n}


def hist_dari_kumulatif(cum_w, cum_n):
    return np.where(cum_n > 0, cum_w / np.maximum(cum_n, 1), np.nan).astype(np.float16)


def modis_z_awal(artefak):
    ka = artefak['kondisi']
    return {'ndvi': ka['ndvi_target'], 'ndwi': ka['ndwi_target'], 'ndwi_valid': ka['ndwi_valid_target']}


def modis_z_dari_mentah(meta, ndvi, ndwi):
    return {'ndvi': z_mentah(meta, 'ndvi', ndvi), 'ndwi': z_mentah(meta, 'ndwi', ndwi),
            'ndwi_valid': np.isfinite(np.asarray(ndwi).astype(np.float16).astype(np.float32)).astype(np.float32)}


def hujan_pola(artefak, nilai):
    ka = artefak['kondisi']
    return {x: ka[f'pola_{x}'].astype(np.float32) * np.float32(nilai[x]) for x in KANAL_HUJAN}


def hujan_klimatologi(meta, tgl):
    k = meta['klimatologi'][str(ke_tanggal(tgl).month)]
    return {x: float(k[x]) for x in KANAL_HUJAN}


def frame_target_rantai(artefak, hist_raw16, rf_mentah, modis_z, gap):
    meta, ka = artefak['meta'], artefak['kondisi']
    ch = meta['frame_ch']
    darat = ka['darat']
    fr = np.zeros((len(ch), *darat.shape), np.float32)
    for x in KANAL_HUJAN:
        fr[ch.index(x)] = z_mentah(meta, x, rf_mentah[x])
    for x in KANAL_MODIS:
        fr[ch.index(x)] = np.asarray(modis_z[x]).astype(np.float32)
    fr[ch.index('land_dist')] = ka['land_dist_target'].astype(np.float32)
    m, s = meta['norm']['gap_days']
    fr[ch.index('gap_days')] = (float(gap) - m) / s
    fr[ch.index('hist_freq')] = z_mentah(meta, 'hist_freq', hist_raw16)
    fr[:, ~darat] = 0
    return fr.astype(np.float16)


def langkah_rantai(artefak, keadaan, gap, rf_mentah, modis_z, lapor=None):
    meta = artefak['meta']
    ch = meta['frame_ch']
    darat = artefak['kondisi']['darat']
    target = frame_target_rantai(artefak, hist_dari_kumulatif(keadaan['cum_w'], keadaan['cum_n']), rf_mentah, modis_z, gap)
    prob = inferensi_bertile(artefak['model'], keadaan['riwayat'], target, int(meta['ukuran_tile']),
                             int(meta['overlap_tile']), lapor)
    air = (prob >= meta['threshold']) & darat
    baru = target.astype(np.float32)
    baru[ch.index('water')] = air.astype(np.float32)
    baru[ch.index('water_valid')] = darat.astype(np.float32)
    baru[ch.index('vv_db')] = np.where(air, meta['rantai']['vv_z_air'], meta['rantai']['vv_z_kering'])
    baru[:, ~darat] = 0
    riwayat = keadaan['riwayat']
    riwayat[:-1] = riwayat[1:]
    riwayat[-1] = baru.astype(np.float16)
    keadaan['cum_w'] = keadaan['cum_w'] + air.astype(np.float32)
    keadaan['cum_n'] = keadaan['cum_n'] + darat.astype(np.float32)
    return prob, air


def keandalan(meta, mode, langkah_ke):
    tabel = meta['keandalan'][mode]
    baris = next((b for b in tabel if b['langkah'] == langkah_ke), None)
    if baris is None:
        return {'label': 'Tidak teruji', 'tingkat': 0,
                'teks': f'{langkah_ke} langkah, di luar rentang uji mundur (maks {tabel[-1]["langkah"]} langkah)'}
    tingkat = 3 if langkah_ke <= 2 else 2 if langkah_ke <= 5 else 1
    label = {3: 'Tinggi', 2: 'Sedang', 1: 'Rendah'}[tingkat]
    return {'label': label, 'tingkat': tingkat, 'F1': baris['F1'], 'F1_persistence': baris['F1_persistence'],
            'teks': f'{langkah_ke} langkah, F1 uji mundur {baris["F1"]:.3f} (Persistence {baris["F1_persistence"]:.3f})'}


def kategori_normal(meta, tgl, luas_km2):
    k = meta['klimatologi'][str(ke_tanggal(tgl).month)]
    if luas_km2 < k['luas_p25']:
        nama = KATEGORI[0]
    elif luas_km2 > k['luas_p75']:
        nama = KATEGORI[2]
    else:
        nama = KATEGORI[1]
    return {'nama': nama, 'p25': float(k['luas_p25']), 'p50': float(k['luas_p50']), 'p75': float(k['luas_p75']),
            'n_scene': int(k['n_scene'])}


def ringkas_hasil(artefak, tgl, prob, air, langkah_ke, mode_keandalan, info):
    meta = artefak['meta']
    darat = artefak['kondisi']['darat']
    n_t, n_d = int(air.sum()), int(darat.sum())
    luas = n_t * meta['px_km2']
    return {'tanggal': ke_tanggal(tgl), 'prob': prob.astype(np.float16), 'air': air,
            'n_tergenang': n_t, 'n_darat': n_d, 'persen_tergenang': n_t / max(n_d, 1) * 100,
            'luas_km2': float(luas),
            'perubahan_km2': float(luas - meta['kondisi_awal_ringkas']['luas_air_km2']),
            'kategori': kategori_normal(meta, tgl, luas), 'langkah_ke': langkah_ke,
            'keandalan': keandalan(meta, mode_keandalan, langkah_ke), **info}


def jalankan_prediksi(artefak, tanggal_urut, penyedia_input, mode_keandalan, lapor_langkah=None, lapor_tile=None):
    rencana = rencana_langkah(artefak['meta'], tanggal_urut)
    keadaan = keadaan_awal(artefak)
    hasil = []
    for i, l in enumerate(rencana):
        masukan = penyedia_input(l)
        if lapor_langkah is not None:
            lapor_langkah(i, len(rencana), l)
        prob, air = langkah_rantai(artefak, keadaan, l['gap'], masukan['rf'], masukan['modis_z'], lapor_tile)
        if l['target']:
            hasil.append(ringkas_hasil(artefak, l['tanggal'], prob, air, l['langkah_ke'], mode_keandalan,
                                       {'hujan': {x: rata_darat(artefak['kondisi']['darat'], masukan['rf'][x])
                                                  for x in KANAL_HUJAN},
                                        'sumber': masukan['sumber'], 'gap_terakhir': l['gap']}))
    del keadaan
    return hasil


def rata_darat(darat, peta):
    nilai = np.asarray(peta, np.float32)[darat]
    nilai = nilai[np.isfinite(nilai)]
    return float(nilai.mean()) if len(nilai) else None


def koordinat_piksel(meta):
    lon_min, lon_maks, lat_min, lat_maks = meta['grid']['extent']
    H, W = meta['grid']['H'], meta['grid']['W']
    lon = lon_min + (np.arange(W) + 0.5) * (lon_maks - lon_min) / W
    lat = lat_maks - (np.arange(H) + 0.5) * (lat_maks - lat_min) / H
    return lon, lat


def faktor_tampilan(H, W, sisi_maks=500):
    return max(1, math.ceil(max(H, W) / sisi_maks))


def _pad(a, f, isi):
    ph, pw = (-a.shape[0]) % f, (-a.shape[1]) % f
    return np.pad(a, ((0, ph), (0, pw)), constant_values=isi) if (ph or pw) else a


def kelas_mayoritas(kelas, f, n_kelas, isi):
    a = _pad(kelas, f, isi)
    h, w = a.shape[0] // f, a.shape[1] // f
    blok = a.reshape(h, f, w, f)
    return np.stack([(blok == i).sum(axis=(1, 3)) for i in range(n_kelas)]).argmax(axis=0).astype(np.uint8)


def rerata_blok(nilai, darat, f):
    v = _pad(np.where(darat, nilai.astype(np.float32), 0), f, 0)
    m = _pad(darat.astype(np.float32), f, 0)
    h, w = v.shape[0] // f, v.shape[1] // f
    jumlah = v.reshape(h, f, w, f).sum(axis=(1, 3))
    n = m.reshape(h, f, w, f).sum(axis=(1, 3))
    return np.where(n > 0, jumlah / np.maximum(n, 1), np.nan).astype(np.float32)


def kelas_genangan(air, darat):
    k = np.full(darat.shape, 2, np.uint8)
    k[darat] = air[darat].astype(np.uint8)
    return k


def kelas_anomali(air, hist_mentah, darat):
    biasa_air = np.nan_to_num(hist_mentah.astype(np.float32), nan=0.0) >= 0.5
    k = np.full(darat.shape, 4, np.uint8)
    k[darat & ~air & ~biasa_air] = 0
    k[darat & air & biasa_air] = 1
    k[darat & air & ~biasa_air] = 2
    k[darat & ~air & biasa_air] = 3
    return k


def kelas_selisih(air_a, air_b, darat):
    k = np.full(darat.shape, 4, np.uint8)
    k[darat & ~air_a & ~air_b] = 0
    k[darat & ~air_a & air_b] = 1
    k[darat & air_a & air_b] = 2
    k[darat & air_a & ~air_b] = 3
    return k


def luas_per_kelas(meta, kelas, n_kelas):
    return [float((kelas == i).sum() * meta['px_km2']) for i in range(n_kelas)]


def koordinat_tampilan(meta, h, w, f):
    lon_min, lon_maks, lat_min, lat_maks = meta['grid']['extent']
    H, W = meta['grid']['H'], meta['grid']['W']
    lon = lon_min + (np.arange(w) + 0.5) * (lon_maks - lon_min) / W * f
    lat = lat_maks - (np.arange(h) + 0.5) * (lat_maks - lat_min) / H * f
    rasio = 1.0 / math.cos(math.radians((lat_min + lat_maks) / 2))
    return lon, lat, rasio


def tabel_hasil(meta, daftar_hasil):
    baris = []
    for h in daftar_hasil:
        k = h['kategori']
        baris.append({
            'tanggal': h['tanggal'].isoformat(), 'luas_tergenang_km2': round(h['luas_km2'], 3),
            'luas_tergenang_ha': round(h['luas_km2'] * 100, 1), 'persen_daratan': round(h['persen_tergenang'], 4),
            'perubahan_vs_observasi_terakhir_km2': round(h['perubahan_km2'], 3), 'kategori': k['nama'],
            'normal_p25_km2': round(k['p25'], 3), 'normal_p75_km2': round(k['p75'], 3),
            'langkah': h['langkah_ke'], 'keandalan': h['keandalan']['label'],
            'hujan_24j_mm': None if h['hujan']['rf24'] is None else round(h['hujan']['rf24'], 2),
            'hujan_72j_mm': None if h['hujan']['rf72'] is None else round(h['hujan']['rf72'], 2),
            'hujan_7h_mm': None if h['hujan']['rf7d'] is None else round(h['hujan']['rf7d'], 2),
            'sumber_hujan': h['sumber'], 'model': meta['nama_model'], 'threshold': meta['threshold'],
        })
    return pd.DataFrame(baris)