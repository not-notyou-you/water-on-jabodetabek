# app.py
import re
import threading
import time
from datetime import timedelta
from pathlib import Path

import streamlit as st

import inti
import tampilan

FOLDER_ARTEFAK = Path(__file__).parent / 'artefak'
IKON_MULAI = ':material/hourglass_top:'
IKON_SUKSES = ':material/check_circle:'
IKON_GAGAL = ':material/error:'

st.set_page_config(page_title='Water on Jabodetabek', layout='centered')
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


@st.cache_resource(show_spinner=False)
def cache_bersama():
    return {'hasil': {}, 'kunci': threading.Lock()}


try:
    with st.spinner('Memuat data...'):
        artefak = ambil_artefak()
except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
    st.error(f'Data aplikasi tidak dapat dimuat: {e}')
    st.stop()

meta = artefak['meta']
darat = artefak['kondisi']['darat']
TGL_AKHIR = inti.tanggal_riwayat_akhir(meta)
GAP_MIN = int(meta['rantai']['gap_min'])
DETIK = float(meta.get('inferensi_cpu_detik', 0.0))
H, W = darat.shape
FAKTOR = inti.faktor_tampilan(H, W)
cocok = re.search(r'(\d{8}) - (\d{8})', meta.get('klimatologi_keterangan', ''))
TGL_AWAL_DATA = inti.ke_tanggal(cocok.group(1)) if cocok else None


def angka(v, desimal=0):
    return f'{v:,.{desimal}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def durasi(detik):
    return f'{detik:.0f} detik' if detik < 90 else f'{detik / 60:.0f} menit'


def hasil_pengamatan():
    air = (artefak['kondisi']['air_terakhir'] == 1) & darat
    luas = float(meta['kondisi_awal_ringkas']['luas_air_km2'])
    return {'tanggal': TGL_AKHIR, 'air': air, 'luas_km2': luas, 'asli': True,
            'kategori': inti.kategori_normal(meta, TGL_AKHIR, luas), 'keandalan': {'label': 'Data asli'}}


def jalankan(tgl):
    kunci = tgl.isoformat()
    bersama = cache_bersama()
    with bersama['kunci']:
        tersimpan = bersama['hasil'].get(kunci)
    if tersimpan is not None:
        st.session_state['hasil'] = tersimpan
        st.toast('Prediksi selesai.', icon=IKON_SUKSES)
        return
    rencana = inti.rencana_langkah(meta, [tgl])
    st.toast(f'Prediksi dimulai, perkiraan {durasi(len(rencana) * DETIK)}.', icon=IKON_MULAI)
    with st.status(f'Menghitung prediksi {inti.format_tanggal(tgl)}...', expanded=True) as status:
        bar = st.progress(0.0)
        posisi = {'i': 0, 'n': len(rencana), 'v': -1}

        def lapor_langkah(i, n, l):
            posisi['i'] = i
            status.update(label=f'Menghitung langkah {i + 1} dari {n}...')

        def lapor_tile(f):
            v = int((posisi['i'] + f) / posisi['n'] * 100)
            if v != posisi['v']:
                posisi['v'] = v
                bar.progress(min(v / 100, 1.0), text=f'{v}%')

        def penyedia(l):
            return {'rf': inti.hujan_pola(artefak, inti.hujan_klimatologi(meta, l['tanggal'])),
                    'modis_z': inti.modis_z_awal(artefak), 'sumber': 'Rata-rata bulanan'}

        try:
            cek = _verifikasi(versi_artefak())
            if not cek['lulus']:
                st.warning('Pemeriksaan awal model tidak lolos. Hasil mungkin berbeda dari penelitian.')
            t0 = time.perf_counter()
            daftar = inti.jalankan_prediksi(artefak, [tgl], penyedia, 'klimatologi', lapor_langkah, lapor_tile)
        except Exception as e:
            status.update(label='Prediksi gagal', state='error', expanded=True)
            st.error(f'Prediksi gagal: {e}')
            st.toast('Prediksi gagal.', icon=IKON_GAGAL)
            return
        status.update(label=f'Selesai dalam {durasi(time.perf_counter() - t0)}', state='complete', expanded=False)
    h = daftar[0]
    hasil = {'tanggal': h['tanggal'], 'air': h['air'], 'luas_km2': h['luas_km2'], 'asli': False,
             'kategori': h['kategori'], 'keandalan': h['keandalan'], 'langkah_ke': h['langkah_ke']}
    st.session_state['hasil'] = hasil
    with bersama['kunci']:
        if len(bersama['hasil']) >= 12:
            bersama['hasil'].pop(next(iter(bersama['hasil'])))
        bersama['hasil'][kunci] = hasil
    st.toast(f'Prediksi selesai: {h["kategori"]["nama"].lower()}.', icon=IKON_SUKSES)


st.markdown(tampilan.kepala_html(), unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="tanya">Mau cek tanggal berapa?</div>', unsafe_allow_html=True)
    kolom_tgl, kolom_tombol = st.columns([2, 1], vertical_alignment='bottom')
    st.session_state.setdefault('tgl_pilih', TGL_AKHIR)
    tgl = kolom_tgl.date_input('Tanggal', min_value=TGL_AKHIR,
                               max_value=TGL_AKHIR + timedelta(days=730), format='DD/MM/YYYY', key='tgl_pilih',
                               label_visibility='collapsed')
    tekan = kolom_tombol.button('Lihat prediksi', type='primary', width='stretch')
    jarak = (tgl - TGL_AKHIR).days
    if 0 < jarak < GAP_MIN:
        st.markdown(tampilan.catatan_html(
            f'Pilih tanggal {inti.format_tanggal(TGL_AKHIR)} (data satelit terakhir) atau mulai '
            f'{inti.format_tanggal(TGL_AKHIR + timedelta(days=GAP_MIN))}.'), unsafe_allow_html=True)

if tekan:
    if jarak == 0:
        st.session_state['hasil'] = hasil_pengamatan()
    elif jarak >= GAP_MIN:
        jalankan(tgl)

hasil = st.session_state.get('hasil') or hasil_pengamatan()
k = hasil['kategori']
yakin = 'Belum teruji' if hasil['keandalan']['label'] == 'Tidak teruji' else hasil['keandalan']['label']
if hasil['asli']:
    ket_yakin = 'pengamatan satelit terakhir, bukan prediksi'
else:
    ket_yakin = f'{hasil["langkah_ke"]} langkah dari data satelit terakhir ({inti.format_tanggal(TGL_AKHIR, True)})'
bulan = inti.NAMA_BULAN[hasil['tanggal'].month - 1]

st.markdown(tampilan.kalimat_html(inti.format_tanggal(hasil['tanggal']), 'tercatat' if hasil['asli'] else 'diperkirakan',
                                  k['nama'], bulan), unsafe_allow_html=True)
st.markdown(tampilan.statistik_html([
    ('Luas genangan', f'{angka(hasil["luas_km2"])} km²', f'sekitar {angka(hasil["luas_km2"] * 100)} hektare',
     tampilan.NAVY),
    (f'Biasanya bulan {bulan}', f'{angka(k["p25"])}–{angka(k["p75"])}',
     f'km², dari data {TGL_AWAL_DATA.year if TGL_AWAL_DATA else ""}–{TGL_AKHIR.year}', tampilan.NAVY),
    ('Seberapa yakin?', yakin, ket_yakin, tampilan.WARNA_YAKIN.get(yakin, tampilan.NAVY)),
]), unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="label-peta">Peta dibanding kondisi biasa</div>', unsafe_allow_html=True)
    penuh = inti.kelas_anomali(hasil['air'], artefak['normal']['hist_mentah'], darat)
    peta = inti.kelas_mayoritas(penuh, FAKTOR, 5, 4)
    lon, lat, rasio = inti.koordinat_tampilan(meta, *peta.shape, FAKTOR)
    st.plotly_chart(tampilan.figur_peta(peta, lon, lat, rasio), width='stretch',
                    config={'displaylogo': False, 'displayModeBar': False, 'scrollZoom': False}, key='peta')
    st.markdown(tampilan.legenda_html(), unsafe_allow_html=True)
    periode_biasa = (f'{inti.NAMA_BULAN[TGL_AWAL_DATA.month - 1]} {TGL_AWAL_DATA.year} – '
                     f'{inti.NAMA_BULAN[TGL_AKHIR.month - 1]} {TGL_AKHIR.year}') if TGL_AWAL_DATA else 'data historis'
    st.markdown(f'<div class="ket-peta">"Biasa" = kondisi tiap lokasi selama {meta["rantai"]["n_scene_hist"]} pengamatan '
                f'satelit, {periode_biasa}. Lokasi dianggap biasa tergenang bila tergenang di minimal separuh '
                'pengamatan.</div>', unsafe_allow_html=True)

with st.expander('Seberapa akurat prediksi ini?'):
    st.markdown(tampilan.akurasi_html(inti.format_tanggal(TGL_AKHIR), meta['keandalan']['klimatologi']),
                unsafe_allow_html=True)

with st.expander('Tentang aplikasi ini'):
    periode = (f'{inti.NAMA_BULAN[TGL_AWAL_DATA.month - 1]} {TGL_AWAL_DATA.year} sampai '
               f'{inti.format_tanggal(TGL_AKHIR)}') if TGL_AWAL_DATA else f'data historis sampai {inti.format_tanggal(TGL_AKHIR)}'
    st.markdown(tampilan.tentang_html(meta['rantai']['n_scene_hist'], periode, meta['rantai']['langkah_hari'],
                                      TGL_AWAL_DATA.year if TGL_AWAL_DATA else '', TGL_AKHIR.year),
                unsafe_allow_html=True)