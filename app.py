# app.py
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import data_nasa
import inti
import tampilan

FOLDER_ARTEFAK = Path(__file__).parent / 'artefak'
ZONA_WIB = timezone(timedelta(hours=7))
MAKS_TANGGAL = 5
MODE_HUJAN = ['Rata-rata bulanan', 'Atur manual', 'Data resmi NASA']
IKON_MULAI = ':material/hourglass_top:'
IKON_SUKSES = ':material/check_circle:'
IKON_GAGAL = ':material/error:'

st.set_page_config(page_title='Water on Jabodetabek', layout='wide')
st.markdown(tampilan.CSS, unsafe_allow_html=True)


def versi_artefak():
    return tuple((p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in sorted(FOLDER_ARTEFAK.iterdir()) if p.is_file())


@st.cache_resource(show_spinner=False, max_entries=1)
def _muat_artefak(versi):
    return inti.muat_artefak(FOLDER_ARTEFAK)


def ambil_artefak():
    return _muat_artefak(versi_artefak())


@st.cache_resource(show_spinner=False, max_entries=1)
def _verifikasi(versi):
    return inti.cek_verifikasi(ambil_artefak(), inti.muat_verify(FOLDER_ARTEFAK))


def verifikasi_awal():
    return _verifikasi(versi_artefak())

@st.cache_resource(show_spinner=False)
def cache_bersama():
    return {'hasil': {}, 'kunci': threading.Lock()}


try:
    with st.spinner('Memuat model dan data riwayat...'):
        artefak = ambil_artefak()
except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
    st.error(f'Artefak tidak dapat dimuat dari folder artefak: {e}')
    st.stop()

meta = artefak['meta']
KUNCI_V3 = ['norm_modis', 'norm_hist', 'klimatologi', 'rantai', 'keandalan', 'akurasi']
kunci_hilang = [k for k in KUNCI_V3 if k not in meta]
if kunci_hilang:
    st.error(f'meta.json di folder artefak masih versi lama (versi kontrak {meta.get("versi_kontrak", "?")}); kunci '
             f'{", ".join(kunci_hilang)} belum ada. Unggah meta.json, normal.npz, dan akurasi.npz terbaru dari '
             'deploy/topik1 hasil chunk ekspor_11 sampai ekspor_15.')
    st.stop()
darat = artefak['kondisi']['darat']
TGL_AKHIR = inti.tanggal_riwayat_akhir(meta)
GAP_MIN = int(meta['rantai']['gap_min'])
DETIK = float(meta.get('inferensi_cpu_detik', 0.0))
H, W = darat.shape
FAKTOR = inti.faktor_tampilan(H, W)


def hari_ini():
    return datetime.now(ZONA_WIB).date()


def rahasia(kunci):
    try:
        return str(st.secrets.get(kunci, ''))
    except Exception:
        return ''


def angka(v, desimal=1):
    return f'{v:,.{desimal}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def durasi(detik):
    if detik < 90:
        return f'{detik:.0f} detik'
    return f'{detik / 60:.1f} menit'.replace('.', ',')


def nilai_slider(kunci, nilai):
    s = meta['slider'][kunci]
    return float(min(max(float(nilai), float(s['min'])), float(s['maks'])))


def terapkan_preset(nama):
    for x in inti.KANAL_HUJAN:
        st.session_state[f'sl_{x}'] = nilai_slider(x, meta['preset'][nama][x])


def tambah_tanggal(tgl=None):
    daftar = st.session_state['daftar_tgl']
    if len(daftar) >= MAKS_TANGGAL:
        return
    if tgl is None:
        tgl = max(daftar) + timedelta(days=int(meta['rantai']['langkah_hari']))
    daftar.append(max(tgl, TGL_AKHIR + timedelta(days=GAP_MIN)))
    for i, d in enumerate(daftar):
        st.session_state[f'tgl_{i}'] = d


def hapus_tanggal():
    daftar = st.session_state['daftar_tgl']
    if len(daftar) > 1:
        daftar.pop()
        st.session_state.pop(f'tgl_{len(daftar)}', None)


st.session_state.setdefault('daftar_tgl', [TGL_AKHIR + timedelta(days=int(meta['rantai']['langkah_hari']))])
for x in inti.KANAL_HUJAN:
    st.session_state.setdefault(f'sl_{x}', nilai_slider(x, meta['slider'][x]['default']))
st.session_state.setdefault('nasa_token', rahasia('NASA_EARTHDATA_TOKEN'))
st.session_state.setdefault('nasa_user', rahasia('NASA_EARTHDATA_USER'))
st.session_state.setdefault('nasa_pass', rahasia('NASA_EARTHDATA_PASSWORD'))
st.session_state.setdefault('gpm_harian', {})
st.session_state.setdefault('modis_cache', {})

teks_riwayat = (f'Observasi Sentinel-1 terakhir: {", ".join(inti.format_tanggal(t, True) for t in meta["tanggal_riwayat"])} '
                f'· luas air {angka(meta["kondisi_awal_ringkas"]["luas_air_km2"])} km²')
st.markdown(tampilan.kepala_html(teks_riwayat), unsafe_allow_html=True)

with st.spinner(f'Memverifikasi model, sekitar {DETIK:.0f} detik pada kunjungan pertama...'):
    hasil_cek = verifikasi_awal()
if not hasil_cek['lulus']:
    st.warning(f'Verifikasi awal tidak lolos (selisih {hasil_cek["selisih_persen"]:.4f} poin, probabilitas '
               f'{hasil_cek["selisih_prob_maks"]:.5f}). Periksa versi torch dan kelengkapan artefak.')


def penyedia_klimatologi(l):
    return {'rf': inti.hujan_pola(artefak, inti.hujan_klimatologi(meta, l['tanggal'])),
            'modis_z': inti.modis_z_awal(artefak), 'sumber': 'Rata-rata bulanan'}


def buat_penyedia(mode, nilai_manual, kred, pakai_modis, catatan, lapor_teks):
    lon_px, lat_px = inti.koordinat_piksel(meta)
    extent = meta['grid']['extent']
    batas_data = hari_ini() - timedelta(days=1)
    sesi = data_nasa.buat_sesi(kred) if mode == MODE_HUJAN[2] else None
    sudah = set()

    def catat(teks):
        if teks not in sudah:
            sudah.add(teks)
            catatan.append(teks)

    def ambil_harian(d):
        simpanan = st.session_state['gpm_harian']
        if d not in simpanan:
            lapor_teks(f'Mengunduh GPM IMERG {inti.format_tanggal(d, True)}')
            simpanan[d] = data_nasa.unduh_imerg_hari(sesi, d, extent)
        return simpanan[d]

    def ambil_modis(d):
        simpanan = st.session_state['modis_cache']
        if d not in simpanan:
            try:
                simpanan[d] = data_nasa.ambil_modis(d, extent, lon_px, lat_px, lapor=lapor_teks)
            except data_nasa.GagalUnduh as e:
                simpanan[d] = None
                catat(f'MODIS {inti.format_tanggal(d, True)} tidak tersedia ({e}); memakai MODIS observasi terakhir.')
        return simpanan[d]

    def penyedia(l):
        d = l['tanggal']
        if mode == MODE_HUJAN[1]:
            if l['target']:
                return {'rf': inti.hujan_pola(artefak, nilai_manual), 'modis_z': inti.modis_z_awal(artefak),
                        'sumber': 'Manual'}
            return penyedia_klimatologi(l)
        if mode == MODE_HUJAN[0]:
            return penyedia_klimatologi(l)
        if d > batas_data:
            catat('Tanggal setelah kemarin belum punya data GPM; langkah tersebut memakai hujan rata-rata bulanan.')
            return {**penyedia_klimatologi(l), 'sumber': 'Rata-rata bulanan (data resmi belum ada)'}
        harian = {d - timedelta(days=i): ambil_harian(d - timedelta(days=i)) for i in range(7)}
        jendela = data_nasa.jendela_hujan(harian, d, lon_px, lat_px)
        if jendela is None:
            catat(f'Sebagian data GPM sekitar {inti.format_tanggal(d, True)} belum terbit; langkah itu memakai hujan '
                  'rata-rata bulanan.')
            return {**penyedia_klimatologi(l), 'sumber': 'Rata-rata bulanan (data resmi belum ada)'}
        modis_z = inti.modis_z_awal(artefak)
        if pakai_modis:
            m = ambil_modis(d)
            if m is not None:
                modis_z = inti.modis_z_dari_mentah(meta, m['ndvi'], m['ndwi'])
        return {'rf': {x: jendela[x] for x in inti.KANAL_HUJAN}, 'modis_z': modis_z,
                'sumber': f'GPM IMERG {"/".join(jendela["produk"])}' + (' + MODIS' if pakai_modis else '')}

    return penyedia


def jalankan(tanggal_urut, mode, nilai_manual, kred, pakai_modis, kunci):
    bersama = cache_bersama()
    if mode != MODE_HUJAN[2]:
        with bersama['kunci']:
            tersimpan = bersama['hasil'].get(kunci)
        if tersimpan is not None:
            st.session_state['hasil_v3'] = tersimpan
            st.toast('Hasil diambil dari prediksi yang sama sebelumnya.', icon=IKON_SUKSES)
            return
    rencana = inti.rencana_langkah(meta, tanggal_urut)
    st.toast(f'Prediksi dimulai: {len(rencana)} langkah, perkiraan {durasi(len(rencana) * DETIK)}.', icon=IKON_MULAI)
    catatan = []
    with st.status(f'Menyiapkan {len(rencana)} langkah prediksi...', expanded=True) as status:
        teks = st.empty()
        bar = st.progress(0.0)
        posisi = {'i': 0, 'n': len(rencana), 'v': -1}

        def lapor_teks(t):
            teks.markdown(f'<div class="kt-sub">{t}</div>', unsafe_allow_html=True)

        def lapor_langkah(i, n, l):
            posisi['i'] = i
            status.update(label=f'Langkah {i + 1} dari {n}: {inti.format_tanggal(l["tanggal"], True)} '
                                f'(jarak {l["gap"]} hari)')
            lapor_teks('Menjalankan model')

        def lapor_tile(f):
            v = int(((posisi['i'] + f) / posisi['n']) * 100)
            if v != posisi['v']:
                posisi['v'] = v
                bar.progress(min(v / 100, 1.0), text=f'{v}%')

        t0 = time.perf_counter()
        try:
            penyedia = buat_penyedia(mode, nilai_manual, kred, pakai_modis, catatan, lapor_teks)
            daftar = inti.jalankan_prediksi(artefak, tanggal_urut, penyedia,
                                            'resmi' if mode == MODE_HUJAN[2] else 'klimatologi',
                                            lapor_langkah, lapor_tile)
        except data_nasa.GagalUnduh as e:
            status.update(label='Pengambilan data NASA gagal', state='error', expanded=True)
            st.error(str(e))
            st.toast('Gagal mengambil data NASA.', icon=IKON_GAGAL)
            return
        except Exception as e:
            status.update(label='Prediksi gagal', state='error', expanded=True)
            st.error(f'Prediksi gagal: {e}')
            st.toast('Prediksi gagal.', icon=IKON_GAGAL)
            return
        detik = time.perf_counter() - t0
        status.update(label=f'Selesai: {len(daftar)} tanggal dalam {durasi(detik)}', state='complete', expanded=False)
    hasil = {'kunci': kunci, 'daftar': daftar, 'catatan': catatan, 'mode': mode, 'detik': detik}
    st.session_state['hasil_v3'] = hasil
    if mode != MODE_HUJAN[2]:
        with bersama['kunci']:
            if len(bersama['hasil']) >= 6:
                bersama['hasil'].pop(next(iter(bersama['hasil'])))
            bersama['hasil'][kunci] = hasil
    ringkas = ', '.join(f'{inti.format_tanggal(h["tanggal"], True)} {h["kategori"]["nama"].lower()}' for h in daftar)
    st.toast(f'Prediksi berhasil: {ringkas}.', icon=IKON_SUKSES)


def judul_tanggal(h):
    return inti.format_tanggal(h['tanggal'], True)


def tampilkan_hasil(hasil):
    daftar = hasil['daftar']
    st.markdown(tampilan.label_kecil(f'Hasil · {hasil["mode"]}'), unsafe_allow_html=True)
    st.markdown('<h3 style="text-align:center;margin:0">Prediksi genangan</h3>', unsafe_allow_html=True)
    for c in hasil['catatan']:
        st.markdown(tampilan.catatan_html(c), unsafe_allow_html=True)
    kartu = []
    for h in daftar:
        k = h['kategori']
        kartu.append({**h, 'judul': judul_tanggal(h), 'luas_teks': angka(h['luas_km2']),
                      'perubahan_teks': angka(abs(h['perubahan_km2'])),
                      'normal_teks': f'{angka(k["p25"])}–{angka(k["p75"])}'})
    st.markdown(tampilan.kartu_tanggal_html(kartu), unsafe_allow_html=True)

    kolom_tren, kolom_jelas = st.columns([1.7, 1], gap='large')
    with kolom_tren:
        st.markdown(tampilan.label_kecil('Luas genangan dibanding rentang normal'), unsafe_allow_html=True)
        normal_bulanan = []
        awal = min([TGL_AKHIR] + [h['tanggal'] for h in daftar])
        akhir = max(h['tanggal'] for h in daftar)
        d = awal
        while d <= akhir + timedelta(days=1):
            k = meta['klimatologi'][str(d.month)]
            normal_bulanan.append((d, k['luas_p25'], k['luas_p75']))
            d += timedelta(days=3)
        st.plotly_chart(tampilan.figur_tren(daftar, (TGL_AKHIR, meta['kondisi_awal_ringkas']['luas_air_km2']),
                                            normal_bulanan), width='stretch',
                        config={'displaylogo': False, 'displayModeBar': False}, key='tren')
    with kolom_jelas:
        st.markdown(tampilan.label_kecil('Cara membaca'), unsafe_allow_html=True)
        st.markdown(tampilan.penjelasan_html(), unsafe_allow_html=True)

    st.markdown(tampilan.label_kecil('Peta per tanggal'), unsafe_allow_html=True)
    pilihan = [judul_tanggal(h) for h in daftar]
    kolom_p1, kolom_p2 = st.columns([1.4, 1])
    terpilih = kolom_p1.radio('Tanggal', pilihan, horizontal=True, key='peta_tgl', label_visibility='collapsed')
    jenis = kolom_p2.radio('Jenis peta', ['Genangan', 'Anomali', 'Probabilitas'], horizontal=True, key='peta_jenis',
                           label_visibility='collapsed')
    h = daftar[pilihan.index(terpilih)] if terpilih in pilihan else daftar[0]
    if jenis == 'Probabilitas':
        peta = inti.rerata_blok(h['prob'], darat, FAKTOR)
        lon, lat, rasio = inti.koordinat_tampilan(meta, *peta.shape, FAKTOR)
        fig = tampilan.figur_probabilitas(peta, lon, lat, rasio)
        rgb = tampilan.rgb_probabilitas(peta)
        legenda = [(tampilan.SKALA_PROB[0][1], 'rendah'), (tampilan.NAVY, 'tinggi'), (tampilan.LAUT, 'laut')]
    else:
        if jenis == 'Anomali':
            penuh = inti.kelas_anomali(h['air'], artefak['normal']['hist_mentah'], darat)
            label, warna = inti.KELAS_ANOMALI, tampilan.WARNA_ANOMALI
        else:
            penuh = inti.kelas_genangan(h['air'], darat)
            label, warna = inti.KELAS_GENANGAN, tampilan.WARNA_BANJIR
        luas = inti.luas_per_kelas(meta, penuh, len(label))
        label_luas = [f'{l} ({angka(v)} km²)' if l != 'Laut' else l for l, v in zip(label, luas)]
        peta = inti.kelas_mayoritas(penuh, FAKTOR, len(label), len(label) - 1)
        lon, lat, rasio = inti.koordinat_tampilan(meta, *peta.shape, FAKTOR)
        fig = tampilan.figur_kelas(peta, lon, lat, rasio, warna, label)
        rgb = tampilan.rgb_kelas(peta, warna)
        legenda = list(zip(warna, label_luas))
    with st.container(border=True):
        st.plotly_chart(fig, width='stretch', config={'displaylogo': False, 'displayModeBar': False}, key='peta_utama')
    st.markdown(tampilan.legenda_html(legenda), unsafe_allow_html=True)
    if jenis == 'Anomali':
        st.caption('Anomali membandingkan prediksi dengan kondisi biasa tiap piksel: "biasa tergenang" berarti '
                   'tergenang pada minimal separuh observasi historis.')
    hujan = ' · '.join(f'{lbl} {angka(h["hujan"][x])} mm' for x, lbl in zip(inti.KANAL_HUJAN, ['24 jam', '72 jam', '7 hari'])
                       if h['hujan'][x] is not None)
    st.caption(f'{inti.format_tanggal(h["tanggal"])} · hujan rata-rata darat {hujan} · sumber {h["sumber"]} · '
               f'{h["keandalan"]["teks"]}. Peta ditampilkan per blok {FAKTOR}×{FAKTOR} piksel; angka dihitung pada '
               'resolusi penuh.')

    if len(daftar) >= 2:
        st.write('')
        st.markdown(tampilan.label_kecil('Bandingkan dua tanggal'), unsafe_allow_html=True)
        kolom_a, kolom_b = st.columns(2)
        a = kolom_a.selectbox('Tanggal A', pilihan, index=0, key='banding_a')
        b = kolom_b.selectbox('Tanggal B', pilihan, index=len(pilihan) - 1, key='banding_b')
        ha, hb = daftar[pilihan.index(a)], daftar[pilihan.index(b)]
        penuh = inti.kelas_selisih(ha['air'], hb['air'], darat)
        luas = inti.luas_per_kelas(meta, penuh, 5)
        st.markdown(tampilan.statistik_html([
            ('Genangan baru di B', f'{angka(luas[1])} km²', f'kering pada {a}'),
            ('Surut di B', f'{angka(luas[3])} km²', f'tergenang pada {a}'),
            ('Selisih luas B − A', f'{"+" if hb["luas_km2"] >= ha["luas_km2"] else "−"}'
                                   f'{angka(abs(hb["luas_km2"] - ha["luas_km2"]))} km²', f'{b} dibanding {a}'),
        ]), unsafe_allow_html=True)
        peta_s = inti.kelas_mayoritas(penuh, FAKTOR, 5, 4)
        lon, lat, rasio = inti.koordinat_tampilan(meta, *peta_s.shape, FAKTOR)
        k1, k2, k3 = st.columns(3, gap='small')
        for kol, hh, judul in ((k1, ha, f'A · {a}'), (k2, hb, f'B · {b}')):
            with kol:
                st.markdown(tampilan.label_kecil(judul), unsafe_allow_html=True)
                kp = inti.kelas_mayoritas(inti.kelas_genangan(hh['air'], darat), FAKTOR, 3, 2)
                with st.container(border=True):
                    st.plotly_chart(tampilan.figur_kelas(kp, lon, lat, rasio, tampilan.WARNA_BANJIR, inti.KELAS_GENANGAN),
                                    width='stretch', config={'displaylogo': False, 'displayModeBar': False},
                                    key=f'banding_{judul}')
        with k3:
            st.markdown(tampilan.label_kecil('Perubahan A → B'), unsafe_allow_html=True)
            with st.container(border=True):
                st.plotly_chart(tampilan.figur_kelas(peta_s, lon, lat, rasio, tampilan.WARNA_PERUBAHAN, inti.KELAS_SELISIH),
                                width='stretch', config={'displaylogo': False, 'displayModeBar': False}, key='banding_selisih')
        st.markdown(tampilan.legenda_html(list(zip(tampilan.WARNA_PERUBAHAN, inti.KELAS_SELISIH))), unsafe_allow_html=True)

    st.write('')
    nama = 'water_on_jabodetabek_' + '_'.join(hh['tanggal'].strftime('%Y%m%d') for hh in daftar)
    png = tampilan.png_laporan(
        [(rgb, f'{jenis} {inti.format_tanggal(h["tanggal"], True)}', [(w, l.replace('²', '2')) for w, l in legenda])],
        f'Water on Jabodetabek - {inti.format_tanggal(h["tanggal"])}',
        f'{h["kategori"]["nama"]} | luas {h["luas_km2"]:.1f} km2 | keandalan {h["keandalan"]["label"].lower()} | '
        f'{meta["nama_model"]}')
    u1, u2 = st.columns(2)
    u1.download_button('Unduh peta PNG (tanggal terpilih)', data=png, file_name=f'{nama}_{jenis.lower()}.png',
                       mime='image/png', width='stretch', icon=':material/image:')
    u2.download_button('Unduh ringkasan CSV (semua tanggal)',
                       data=inti.tabel_hasil(meta, daftar).to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'{nama}.csv', mime='text/csv', width='stretch', icon=':material/table_view:')


tab_prediksi, tab_akurasi, tab_tentang = st.tabs(['Prediksi', 'Akurasi model', 'Tentang'])

with tab_prediksi:
    with st.container(border=True):
        st.markdown(tampilan.label_kecil(f'Tanggal prediksi (maksimal {MAKS_TANGGAL})'), unsafe_allow_html=True)
        daftar_tgl = st.session_state['daftar_tgl']
        kolom_tgl = st.columns(MAKS_TANGGAL)
        for i in range(len(daftar_tgl)):
            st.session_state.setdefault(f'tgl_{i}', daftar_tgl[i])
            daftar_tgl[i] = kolom_tgl[i].date_input(f'Tanggal {i + 1}', key=f'tgl_{i}',
                                                    min_value=TGL_AKHIR + timedelta(days=GAP_MIN),
                                                    max_value=TGL_AKHIR + timedelta(days=730), format='DD/MM/YYYY')
        b1, b2, b3, _ = st.columns([1, 1, 1, 1.4])
        b1.button('Tambah tanggal', on_click=tambah_tanggal, width='stretch', icon=':material/add:',
                  disabled=len(daftar_tgl) >= MAKS_TANGGAL)
        b2.button('Tambah hari ini', on_click=tambah_tanggal, args=(hari_ini(),), width='stretch',
                  icon=':material/today:', disabled=len(daftar_tgl) >= MAKS_TANGGAL)
        b3.button('Hapus terakhir', on_click=hapus_tanggal, width='stretch', icon=':material/remove:',
                  disabled=len(daftar_tgl) <= 1)
        tanggal_urut, masalah = inti.periksa_tanggal(meta, daftar_tgl)
        for m in masalah:
            st.markdown(tampilan.catatan_html(m), unsafe_allow_html=True)
        if not masalah:
            rencana = inti.rencana_langkah(meta, tanggal_urut)
            terjauh = max(r['langkah_ke'] for r in rencana)
            batas_uji = meta['keandalan']['klimatologi'][-1]['langkah']
            st.markdown(f'<div class="info-tanggal">{tampilan.ikon_ui("kalender")}{len(rencana)} langkah prediksi '
                        f'berantai dari {inti.format_tanggal(TGL_AKHIR, True)} · perkiraan waktu '
                        f'{durasi(len(rencana) * DETIK)}</div>', unsafe_allow_html=True)
            if terjauh > batas_uji:
                st.markdown(tampilan.catatan_html(
                    f'Tanggal terjauh butuh {terjauh} langkah, melebihi rentang uji mundur ({batas_uji} langkah). '
                    'Hasilnya tetap dihitung, tetapi keandalannya belum teruji.'), unsafe_allow_html=True)

        st.markdown(tampilan.label_kecil('Sumber curah hujan'), unsafe_allow_html=True)
        mode = st.radio('Sumber curah hujan', MODE_HUJAN, horizontal=True, key='mode_hujan',
                        label_visibility='collapsed')
        nilai_manual = None
        kred = None
        pakai_modis = False
        if mode == MODE_HUJAN[0]:
            st.caption('Setiap langkah memakai rata-rata hujan historis bulan yang bersangkutan. Tidak perlu login.')
        elif mode == MODE_HUJAN[1]:
            st.caption('Nilai slider dipakai pada langkah terakhir menuju setiap tanggal; langkah antara memakai '
                       'rata-rata bulanan. Pola sebaran hujan mengikuti observasi terakhir.')
            kp = st.columns(len(meta['preset']))
            for kol, nama in zip(kp, meta['preset']):
                kol.button(nama, on_click=terapkan_preset, args=(nama,), width='stretch', key=f'preset_{nama}')
            ks = st.columns(3, gap='large')
            for kol, x in zip(ks, inti.KANAL_HUJAN):
                s = meta['slider'][x]
                kol.slider(f'{s["label"]} (mm)', min_value=float(s['min']), max_value=float(s['maks']),
                           step=float(s['langkah']), key=f'sl_{x}')
                kol.markdown(f'<div class="rentang">Rentang historis {s["min"]}–{s["maks"]} mm</div>',
                             unsafe_allow_html=True)
            nilai_manual = {x: st.session_state[f'sl_{x}'] for x in inti.KANAL_HUJAN}
        else:
            st.caption('Hujan GPM IMERG harian dari NASA untuk setiap langkah hingga kemarin; langkah setelahnya '
                       'memakai rata-rata bulanan. Membutuhkan akun NASA Earthdata.')
            kred = {'token': st.session_state['nasa_token'], 'user': st.session_state['nasa_user'],
                    'password': st.session_state['nasa_pass']}
            with st.expander('Akses NASA Earthdata', expanded=not data_nasa.kredensial_lengkap(kred), icon=':material/key:'):
                st.caption('Isi token, atau username dan password. Kredensial hanya disimpan di sesi browser ini.')
                st.text_input('NASA_EARTHDATA_TOKEN', type='password', key='nasa_token')
                ku, kpw = st.columns(2)
                ku.text_input('NASA_EARTHDATA_USER', key='nasa_user')
                kpw.text_input('NASA_EARTHDATA_PASSWORD', type='password', key='nasa_pass')
            kred = {'token': st.session_state['nasa_token'], 'user': st.session_state['nasa_user'],
                    'password': st.session_state['nasa_pass']}
            pakai_modis = st.checkbox('Gunakan juga MODIS terbaru (komposit 8 harian, layanan ORNL)', value=False,
                                      key='pakai_modis')

        kunci = (tuple(d.isoformat() for d in tanggal_urut), mode,
                 tuple(sorted(nilai_manual.items())) if nilai_manual else None, pakai_modis if kred else None)
        siap = not masalah and (mode != MODE_HUJAN[2] or data_nasa.kredensial_lengkap(kred))
        _, kt, _ = st.columns([1, 1.2, 1])
        tekan = kt.button('Jalankan prediksi', type='primary', width='stretch', key='tombol_prediksi',
                          icon=':material/play_arrow:', disabled=not siap)
        if mode == MODE_HUJAN[2] and not data_nasa.kredensial_lengkap(kred):
            st.caption('Isi kredensial NASA Earthdata untuk mengaktifkan tombol.')
        tempat_peringatan = st.empty()
    if tekan:
        jalankan(tanggal_urut, mode, nilai_manual, kred, pakai_modis, kunci)
    lama = st.session_state.get('hasil_v3')
    if lama and lama['kunci'] != kunci:
        tempat_peringatan.markdown(tampilan.catatan_html('Input berubah sejak prediksi terakhir. Tekan Jalankan '
                                                         'prediksi untuk memperbarui hasil.'), unsafe_allow_html=True)
    st.write('')
    if st.session_state.get('hasil_v3'):
        tampilkan_hasil(st.session_state['hasil_v3'])
    else:
        st.markdown(f'<div class="kosong">{tampilan.ikon_ui("info")} Pilih tanggal dan sumber hujan, lalu tekan '
                    '<b>Jalankan prediksi</b>.</div>', unsafe_allow_html=True)

with tab_akurasi:
    st.markdown(tampilan.label_kecil('Keandalan per jumlah langkah'), unsafe_allow_html=True)
    k1, k2 = st.columns([1.6, 1], gap='large')
    with k1:
        st.plotly_chart(tampilan.figur_keandalan(meta['keandalan']), width='stretch',
                        config={'displaylogo': False, 'displayModeBar': False}, key='keandalan')
    with k2:
        st.markdown(tampilan.konteks_html([
            'Uji mundur: prediksi berantai dijalankan dari setiap scene test lalu dibandingkan dengan genangan '
            'aktual hasil SAR pada observasi-observasi berikutnya.',
            'Garis putus-putus adalah Persistence (menganggap genangan tidak berubah sejak awal). Model berguna '
            'selama garisnya di atas Persistence.',
            f'<span class="kt-sub">{meta["keandalan_keterangan"]}</span>']), unsafe_allow_html=True)
    df_k = pd.DataFrame(meta['keandalan']['resmi'])[['langkah', 'hari_rata', 'F1', 'F1_persistence', 'MAE_luas_km2']]
    df_k.columns = ['Langkah', 'Rata-rata hari', 'F1 model', 'F1 Persistence', 'Galat luas (km²)']
    st.dataframe(df_k.round(3), hide_index=True, width='stretch')

    if artefak['akurasi'] and meta.get('akurasi'):
        st.write('')
        st.markdown(tampilan.label_kecil('Prediksi vs aktual pada scene test (satu langkah)'), unsafe_allow_html=True)
        scene = meta['akurasi']['scene']
        nama_scene = [inti.format_tanggal(s['tanggal'], True) for s in scene]
        pilih = st.selectbox('Scene test', nama_scene, index=len(nama_scene) - 1, key='galeri')
        s = scene[nama_scene.index(pilih)]
        st.markdown(tampilan.statistik_html([
            ('F1', angka(s['F1'], 3), f'recall {angka(s["Recall"], 3)} · precision {angka(s["Precision"], 3)}'),
            ('Luas prediksi', f'{angka(s["luas_pred_km2"])} km²', 'piksel dengan data SAR'),
            ('Luas aktual (SAR)', f'{angka(s["luas_aktual_km2"])} km²', 'ambang VV −15 dB'),
        ]), unsafe_allow_html=True)
        fa = int(meta['akurasi']['faktor_tampilan'])
        g1, g2 = st.columns(2, gap='medium')
        for kol, jenis_g, judul in ((g1, 'prediksi', 'Prediksi model'), (g2, 'aktual', 'Aktual (SAR)')):
            peta = artefak['akurasi'][f'{jenis_g}_{s["tanggal"]}']
            lon, lat, rasio = inti.koordinat_tampilan(meta, *peta.shape, fa)
            with kol:
                st.markdown(tampilan.label_kecil(judul), unsafe_allow_html=True)
                with st.container(border=True):
                    st.plotly_chart(tampilan.figur_kelas(peta, lon, lat, rasio, tampilan.WARNA_BANJIR,
                                                         ['Kering', 'Tergenang', 'Laut / tanpa data']),
                                    width='stretch', config={'displaylogo': False, 'displayModeBar': False},
                                    key=f'galeri_{jenis_g}')
        st.markdown(tampilan.legenda_html(list(zip(tampilan.WARNA_BANJIR, ['kering', 'tergenang', 'laut / tanpa data']))),
                    unsafe_allow_html=True)
        st.plotly_chart(tampilan.figur_akurasi_luas(scene), width='stretch',
                        config={'displaylogo': False, 'displayModeBar': False}, key='akurasi_luas')

with tab_tentang:
    st.markdown(f'**{meta["nama_model"]}** ({meta["jenis_model"]}) memprediksi genangan per piksel (sekitar 50 m) di '
                'Jabodetabek pada observasi Sentinel-1 berikutnya dari data multisensor: Sentinel-1 (radar), '
                'MODIS (NDVI, NDWI), dan GPM IMERG (curah hujan).')
    st.markdown(
        '**Cara kerja**\n'
        f'- Model melihat {meta["seq_len"] - 1} observasi Sentinel-1 terakhir dan kondisi target (hujan, MODIS, jarak '
        'hari, frekuensi genangan historis), lalu memberi probabilitas tergenang untuk setiap piksel.\n'
        f'- Piksel dianggap tergenang bila probabilitasnya ≥ {meta["threshold"]:.3f} (threshold F1 terbaik pada data '
        'validasi).\n'
        f'- Untuk tanggal yang jauh, prediksi dibuat berantai per sekitar {meta["rantai"]["langkah_hari"]} hari. Hasil '
        'tiap langkah menjadi riwayat untuk langkah berikutnya.\n'
        '- Kategori normal membandingkan luas genangan dengan persentil 25–75 luas historis pada bulan yang sama.')
    st.markdown('**Perbandingan model pada data test**')
    df_b = pd.DataFrame(meta['pembanding']).rename(columns={'nama': 'Model', 'jenis': 'Jenis',
                                                            'PR_AUC_dyn': 'PR-AUC dinamis', 'Infer_s': 'Inferensi (s)'})
    st.dataframe(df_b.round(4), hide_index=True, width='stretch')
    mt = meta['metrik_test']
    st.markdown(f'Metrik utama {meta["nama_model"]}: PR-AUC piksel dinamis {mt["PR_AUC_dyn_CI"]}, F1 {mt["F1"]:.3f}, '
                f'recall {mt["Recall"]:.3f}, precision {mt["Precision"]:.3f}.')
    st.markdown(
        '**Keterbatasan**\n'
        f'- Riwayat radar berhenti pada {inti.format_tanggal(TGL_AKHIR)}. Setelahnya, kondisi permukaan hanya diperkirakan '
        'oleh model, sehingga keandalan menurun dengan jumlah langkah (lihat tab Akurasi model).\n'
        '- Model belajar terutama dari riwayat genangan dan frekuensi historis; pengaruh curah hujan kecil. Perbedaan '
        'antar skenario hujan yang tipis adalah perilaku model, bukan kesalahan aplikasi.\n'
        '- Label air berasal dari ambang radar VV −15 dB: dapat tertukar dengan permukaan halus lain dan sulit '
        'mendeteksi genangan di bawah vegetasi atau di antara bangunan.\n'
        '- Uji mundur hanya mencakup periode Maret–Juni 2026.\n'
        '- Hasil adalah estimasi riset, bukan peringatan resmi kebencanaan. Untuk informasi resmi, rujuk BMKG dan BPBD.')