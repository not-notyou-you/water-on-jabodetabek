# topik1/inti.py
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

KANAL_HUJAN = ('rf24', 'rf72', 'rf7d')
KANAL_TETAP = (('ndvi', 'ndvi_target'), ('ndwi', 'ndwi_target'), ('ndwi_valid', 'ndwi_valid_target'),
               ('land_dist', 'land_dist_target'), ('hist_freq', 'hist_target'))
NAMA_BULAN = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus',
              'September', 'Oktober', 'November', 'Desember']
KELAS_BANJIR = ['Tidak tergenang', 'Tergenang', 'Laut']
KELAS_PERUBAHAN = ['Tetap kering', 'Genangan baru', 'Tetap tergenang', 'Surut', 'Laut / tanpa data']
JUMLAH_THREAD = 2


def muat_meta(folder):
    with open(Path(folder) / 'meta.json', encoding='utf-8') as f:
        return json.load(f)


def muat_verify(folder):
    with open(Path(folder) / 'verify.json', encoding='utf-8') as f:
        return json.load(f)


def muat_artefak(folder):
    folder = Path(folder)
    meta = muat_meta(folder)
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
                    if k not in ch:
                        raise ValueError(f'Kanal {k} pada {p.name} tidak ada di frame_ch')
                    if k in terisi:
                        raise ValueError(f'Kanal {k} muncul ganda pada frame {t}')
                    riwayat[t, ch.index(k)] = d[k]
                    terisi.add(k)
        if terisi != set(ch):
            raise ValueError(f'Kanal frame {t} tidak lengkap: {sorted(set(ch) - terisi)}')
    model = torch.jit.load(str(folder / 'model.ts'), map_location='cpu').eval()
    return {'meta': meta, 'kondisi': kondisi, 'riwayat': riwayat, 'model': model}


def bentuk_frame_target(artefak, skenario):
    meta, ka = artefak['meta'], artefak['kondisi']
    ch = meta['frame_ch']
    darat = ka['darat']
    target = np.zeros((len(ch), *darat.shape), np.float16)

    def isi(kanal, nilai):
        a = np.asarray(nilai, np.float32)
        if a.ndim == 0:
            a = np.full(darat.shape, a, np.float32)
        a = a.copy()
        a[~darat] = 0
        target[ch.index(kanal)] = a.astype(np.float16)

    for x in KANAL_HUJAN:
        m, s = meta['norm'][x]
        isi(x, (ka[f'pola_{x}'].astype(np.float32) * np.float32(skenario[x]) - np.float32(m)) / np.float32(s))
    for kanal, kunci in KANAL_TETAP:
        isi(kanal, ka[kunci].astype(np.float32))
    m, s = meta['norm']['gap_days']
    isi('gap_days', (np.float32(skenario['gap_days']) - np.float32(m)) / np.float32(s))
    return target


def inferensi_bertile(model, riwayat, target, ukuran_tile, overlap_tile):
    torch.set_num_threads(JUMLAH_THREAD)
    T = riwayat.shape[0] + 1
    C, H, W = target.shape
    akumulasi = np.zeros((H, W), np.float32)
    hitung = np.zeros((H, W), np.float32)
    stride = ukuran_tile - 2 * overlap_tile
    x = np.zeros((T, C, ukuran_tile, ukuran_tile), np.float32)
    with torch.inference_mode():
        for r in range(0, H, stride):
            for c in range(0, W, stride):
                r0, c0 = max(r - overlap_tile, 0), max(c - overlap_tile, 0)
                r1, c1 = min(r0 + ukuran_tile, H), min(c0 + ukuran_tile, W)
                h, w = r1 - r0, c1 - c0
                x.fill(0)
                x[:T - 1, :, :h, :w] = riwayat[:, :, r0:r1, c0:c1]
                x[T - 1, :, :h, :w] = target[:, r0:r1, c0:c1]
                p = torch.sigmoid(model(torch.from_numpy(x)[None]))[0].numpy()
                akumulasi[r0:r1, c0:c1] += p[:h, :w]
                hitung[r0:r1, c0:c1] += 1
    return akumulasi / np.maximum(hitung, 1)


def status_genangan(meta, persen):
    if persen < meta['status']['waspada']:
        return 'Aman'
    if persen < meta['status']['bahaya']:
        return 'Waspada'
    return 'Bahaya'


def ringkasan(meta, darat, prob):
    air = (prob >= meta['threshold']) & darat
    n_tergenang, n_darat = int(air.sum()), int(darat.sum())
    persen = n_tergenang / max(n_darat, 1) * 100
    luas = n_tergenang * meta['px_km2']
    return {'air': air, 'n_tergenang': n_tergenang, 'n_darat': n_darat,
            'persen_tergenang': float(persen), 'luas_tergenang_km2': float(luas),
            'perubahan_km2': float(luas - meta['kondisi_awal_ringkas']['luas_air_km2']),
            'prob_rata_darat': float(prob[darat].mean()), 'status': status_genangan(meta, persen)}


def prediksi_skenario(artefak, skenario, ukuran_tile=None):
    meta = artefak['meta']
    darat = artefak['kondisi']['darat']
    tile = int(ukuran_tile or meta['ukuran_tile'])
    target = bentuk_frame_target(artefak, skenario)
    prob = inferensi_bertile(artefak['model'], artefak['riwayat'], target, tile, int(meta['overlap_tile']))
    return {'prob': prob, **ringkasan(meta, darat, prob)}


def untuk_cache(hasil):
    return {**hasil, 'prob': hasil['prob'].astype(np.float16)}


def kunci_skenario(skenario, urutan):
    return tuple((k, round(float(skenario[k]), 4)) for k in urutan)


def cek_verifikasi(verify, hasil):
    idx = np.asarray(verify['indeks_sampel'], np.int64)
    prob = hasil['prob'].ravel()[idx].astype(np.float32)
    sel_persen = abs(hasil['persen_tergenang'] - verify['persen_tergenang'])
    sel_prob = float(np.max(np.abs(prob - np.asarray(verify['prob_sampel'], np.float32)))) if len(idx) else 0.0
    return {'lulus': sel_persen <= verify['toleransi_persen'] and sel_prob <= verify['toleransi_prob'],
            'selisih_persen': float(sel_persen), 'selisih_prob_maks': sel_prob,
            'toleransi_persen': verify['toleransi_persen'], 'toleransi_prob': verify['toleransi_prob']}


def ke_tanggal(teks):
    return datetime.strptime(str(teks), '%Y%m%d')


def tanggal_prediksi(meta, gap_days):
    return ke_tanggal(meta['tanggal_riwayat'][-1]) + timedelta(days=int(round(float(gap_days))))


def format_tanggal(tgl):
    if not isinstance(tgl, datetime):
        tgl = ke_tanggal(tgl)
    return f'{tgl.day} {NAMA_BULAN[tgl.month - 1]} {tgl.year}'


def faktor_tampilan(H, W, sisi_maks=500):
    return max(1, math.ceil(max(H, W) / sisi_maks))


def _pad(a, f, isi):
    H, W = a.shape
    ph, pw = (-H) % f, (-W) % f
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)), constant_values=isi)
    return a


def kelas_mayoritas(kelas, f, n_kelas, isi):
    a = _pad(kelas, f, isi)
    h, w = a.shape[0] // f, a.shape[1] // f
    blok = a.reshape(h, f, w, f)
    hitung = np.stack([(blok == i).sum(axis=(1, 3)) for i in range(n_kelas)])
    return hitung.argmax(axis=0).astype(np.uint8)


def rerata_blok(nilai, darat, f):
    v = _pad(np.where(darat, nilai.astype(np.float32), 0), f, 0)
    m = _pad(darat.astype(np.float32), f, 0)
    h, w = v.shape[0] // f, v.shape[1] // f
    jumlah = v.reshape(h, f, w, f).sum(axis=(1, 3))
    n = m.reshape(h, f, w, f).sum(axis=(1, 3))
    return np.where(n > 0, jumlah / np.maximum(n, 1), np.nan).astype(np.float32)


def kelas_banjir(air, darat):
    k = np.full(darat.shape, 2, np.uint8)
    k[darat] = air[darat].astype(np.uint8)
    return k


def kelas_perubahan(air, air_terakhir, darat):
    k = np.full(darat.shape, 4, np.uint8)
    awal_kering = darat & (air_terakhir == 0)
    awal_air = darat & (air_terakhir == 1)
    k[awal_kering & ~air] = 0
    k[awal_kering & air] = 1
    k[awal_air & air] = 2
    k[awal_air & ~air] = 3
    return k


def luas_perubahan(meta, kelas):
    return {nama: float((kelas == i).sum() * meta['px_km2']) for i, nama in enumerate(KELAS_PERUBAHAN[:4])}


def koordinat_tampilan(meta, h, w, f):
    lon_min, lon_maks, lat_min, lat_maks = meta['grid']['extent']
    H, W = meta['grid']['H'], meta['grid']['W']
    dlon = (lon_maks - lon_min) / W * f
    dlat = (lat_maks - lat_min) / H * f
    lon = lon_min + (np.arange(w) + 0.5) * dlon
    lat = lat_maks - (np.arange(h) + 0.5) * dlat
    rasio = 1.0 / math.cos(math.radians((lat_min + lat_maks) / 2))
    return lon, lat, rasio


def tabel_ringkasan(meta, skenario, hasil):
    baris = [('Model', 'Nama', meta['nama_model']),
             ('Model', 'Threshold', meta['threshold']),
             ('Riwayat', 'Tanggal observasi', ', '.join(meta['tanggal_riwayat'])),
             ('Skenario', 'Tanggal prediksi', tanggal_prediksi(meta, skenario['gap_days']).strftime('%Y-%m-%d'))]
    for k, s in meta['slider'].items():
        baris.append(('Skenario', f'{s["label"]} ({s["satuan"]})', skenario[k]))
    baris += [('Hasil', 'Persen tergenang (%)', round(hasil['persen_tergenang'], 4)),
              ('Hasil', 'Luas tergenang (km2)', round(hasil['luas_tergenang_km2'], 4)),
              ('Hasil', 'Luas tergenang (ha)', round(hasil['luas_tergenang_km2'] * 100, 2)),
              ('Hasil', 'Perubahan vs scene terakhir (km2)', round(hasil['perubahan_km2'], 4)),
              ('Hasil', 'Probabilitas rata-rata darat', round(hasil['prob_rata_darat'], 6)),
              ('Hasil', 'Status', hasil['status']),
              ('Ambang', 'Waspada mulai (%)', round(meta['status']['waspada'], 4)),
              ('Ambang', 'Bahaya mulai (%)', round(meta['status']['bahaya'], 4))]
    return pd.DataFrame(baris, columns=['Bagian', 'Parameter', 'Nilai'])