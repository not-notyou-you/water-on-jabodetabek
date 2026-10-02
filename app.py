# topik1/app.py
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import inti
import tampilan

FOLDER_ARTEFAK = Path(__file__).parent / 'artefak'
MODE_PETA = ['Tergenang / tidak', 'Probabilitas', 'Perubahan']

st.set_page_config(page_title='Simulasi Genangan Jabodetabek', layout='wide')
st.markdown(tampilan.CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def ambil_artefak():
    return inti.muat_artefak(FOLDER_ARTEFAK)


try:
    with st.spinner('Memuat model dan data riwayat...'):
        artefak = ambil_artefak()
except (FileNotFoundError, ValueError, RuntimeError) as e:
    st.error(f'Artefak tidak dapat dimuat dari folder topik1/artefak: {e}')
    st.stop()

meta = artefak['meta']
darat = artefak['kondisi']['darat']
URUTAN = list(meta['slider'])
DETIK = float(meta.get('inferensi_cpu_detik', 0.0))


@st.cache_data(max_entries=8, show_spinner=False)
def prediksi_tersimpan(kunci):
    return inti.untuk_cache(inti.prediksi_skenario(ambil_artefak(), dict(kunci)))


@st.cache_resource(show_spinner=False)
def verifikasi_awal():
    verify = inti.muat_verify(FOLDER_ARTEFAK)
    hasil = inti.prediksi_skenario(ambil_artefak(), verify['skenario'])
    return inti.cek_verifikasi(verify, hasil)


def nilai_slider(kunci, nilai):
    s = meta['slider'][kunci]
    v = min(max(float(nilai), float(s['min'])), float(s['maks']))
    return int(round(v)) if isinstance(s['langkah'], int) else float(v)


def terapkan_preset(nama):
    for k, v in meta['preset'][nama].items():
        if k in meta['slider']:
            st.session_state[f'sl_{k}'] = nilai_slider(k, v)


for k, s in meta['slider'].items():
    st.session_state.setdefault(f'sl_{k}', nilai_slider(k, s['default']))

st.title('Simulasi Skenario Genangan Jabodetabek')
st.markdown(
    f'<div class="baris-info">Atur skenario curah hujan dan jarak hari, lalu model {meta["nama_model"]} '
    'memprediksi peta genangan pada observasi Sentinel-1 berikutnya.</div>',
    unsafe_allow_html=True,
)
tanggal_riwayat = ', '.join(inti.format_tanggal(t) for t in meta['tanggal_riwayat'])
ringkas_awal = meta['kondisi_awal_ringkas']
st.markdown(
    f'<div class="baris-info">{tampilan.ikon_ui("satelit")}Riwayat: observasi Sentinel-1 tanggal {tanggal_riwayat}, '
    f'luas air terakhir {ringkas_awal["luas_air_km2"]:.2f} km² ({ringkas_awal["persen_tergenang"]:.2f}%).</div>',
    unsafe_allow_html=True,
)

with st.spinner(f'Menyiapkan verifikasi awal model, sekitar {DETIK:.0f} detik pada kunjungan pertama...'):
    hasil_cek = verifikasi_awal()
if not hasil_cek['lulus']:
    st.warning(
        'Verifikasi awal tidak lolos: hasil aplikasi berbeda dari acuan notebook '
        f'(selisih persen {hasil_cek["selisih_persen"]:.4f} poin, toleransi {hasil_cek["toleransi_persen"]}; '
        f'selisih probabilitas maks {hasil_cek["selisih_prob_maks"]:.5f}, toleransi {hasil_cek["toleransi_prob"]}). '
        'Periksa versi torch dan kelengkapan artefak. Aplikasi tetap berjalan.'
    )

kolom_input, kolom_hasil = st.columns([1, 2.3], gap='large')

with kolom_input:
    st.subheader('Skenario')
    st.caption('Preset')
    kolom_preset = st.columns(len(meta['preset']))
    for kol, nama in zip(kolom_preset, meta['preset']):
        kol.button(nama, on_click=terapkan_preset, args=(nama,), width='stretch', key=f'preset_{nama}')
    for k, s in meta['slider'].items():
        bulat = isinstance(s['langkah'], int)
        konversi = int if bulat else float
        st.slider(f'{s["label"]} ({s["satuan"]})', min_value=konversi(s['min']), max_value=konversi(s['maks']),
                  step=konversi(s['langkah']), key=f'sl_{k}')
        st.markdown(f'<div class="rentang">Rentang historis: {s["min"]}–{s["maks"]} {s["satuan"]}</div>',
                    unsafe_allow_html=True)
    skenario = {k: st.session_state[f'sl_{k}'] for k in URUTAN}
    kunci = inti.kunci_skenario(skenario, URUTAN)
    tgl_pred = inti.tanggal_prediksi(meta, skenario['gap_days'])
    st.markdown(f'<div class="baris-info">{tampilan.ikon_ui("kalender")}Tanggal prediksi: '
                f'<b>{inti.format_tanggal(tgl_pred)}</b></div>', unsafe_allow_html=True)
    if st.button('Prediksi', type='primary', width='stretch'):
        with st.spinner(f'Menjalankan model, perkiraan sekitar {DETIK:.0f} detik...'):
            t0 = time.perf_counter()
            hasil = prediksi_tersimpan(kunci)
            st.session_state['prediksi'] = {'kunci': kunci, 'skenario': dict(skenario), 'hasil': hasil,
                                            'detik': time.perf_counter() - t0}
    tersimpan = st.session_state.get('prediksi')
    if tersimpan and tersimpan['kunci'] != kunci:
        st.warning('Skenario berubah, tekan Prediksi untuk memperbarui hasil.')

with kolom_hasil:
    tersimpan = st.session_state.get('prediksi')
    if not tersimpan:
        st.markdown(f'<div class="kosong">{tampilan.ikon_ui("info", "#6b7280")}'
                    'Pilih preset atau atur slider, lalu tekan <b>Prediksi</b>.</div>', unsafe_allow_html=True)
    else:
        hasil = tersimpan['hasil']
        sk = tersimpan['skenario']
        tgl_hasil = inti.format_tanggal(inti.tanggal_prediksi(meta, sk['gap_days']))
        st.subheader(f'Hasil prediksi {tgl_hasil}')
        st.caption(' | '.join(f'{meta["slider"][k]["label"]}: {sk[k]} {meta["slider"][k]["satuan"]}' for k in URUTAN)
                   + f' | waktu proses {tersimpan["detik"]:.1f} detik')

        H, W = darat.shape
        f = inti.faktor_tampilan(H, W)
        mode = st.radio('Tampilan peta', MODE_PETA, horizontal=True, key='mode_peta', label_visibility='collapsed')
        if mode == 'Probabilitas':
            peta = inti.rerata_blok(hasil['prob'], darat, f)
            lon, lat, rasio = inti.koordinat_tampilan(meta, *peta.shape, f)
            fig = tampilan.figur_probabilitas(peta, lon, lat, rasio, meta['threshold'], 'Probabilitas tergenang')
            rgb = tampilan.rgb_probabilitas(peta)
            legenda = [('#f0f9ff', 'Probabilitas 0'), ('#60a5fa', 'Probabilitas 0.5'),
                       ('#1e3a8a', 'Probabilitas 1'), ('#e5e7eb', 'Laut')]
        elif mode == 'Perubahan':
            kelas_penuh = inti.kelas_perubahan(hasil['air'], artefak['kondisi']['air_terakhir'], darat)
            peta = inti.kelas_mayoritas(kelas_penuh, f, len(inti.KELAS_PERUBAHAN), 4)
            lon, lat, rasio = inti.koordinat_tampilan(meta, *peta.shape, f)
            luas = inti.luas_perubahan(meta, kelas_penuh)
            label = [f'{n} ({luas[n]:.1f} km²)' if n in luas else n for n in inti.KELAS_PERUBAHAN]
            fig = tampilan.figur_kelas(peta, lon, lat, rasio, tampilan.WARNA_PERUBAHAN, label,
                                       f'Perubahan dibanding observasi {inti.format_tanggal(ringkas_awal["tanggal"])}')
            rgb = tampilan.rgb_kelas(peta, tampilan.WARNA_PERUBAHAN)
            legenda = list(zip(tampilan.WARNA_PERUBAHAN, label))
        else:
            peta = inti.kelas_mayoritas(inti.kelas_banjir(hasil['air'], darat), f, len(inti.KELAS_BANJIR), 2)
            lon, lat, rasio = inti.koordinat_tampilan(meta, *peta.shape, f)
            fig = tampilan.figur_kelas(peta, lon, lat, rasio, tampilan.WARNA_BANJIR, inti.KELAS_BANJIR,
                                       'Peta tergenang dan tidak tergenang')
            rgb = tampilan.rgb_kelas(peta, tampilan.WARNA_BANJIR)
            legenda = list(zip(tampilan.WARNA_BANJIR, inti.KELAS_BANJIR))
        st.plotly_chart(fig, width='stretch', config={'displaylogo': False})
        if f > 1:
            st.caption(f'Peta ditampilkan per blok {f}×{f} piksel. Semua angka dihitung pada resolusi penuh.')

        kolom_donat, kolom_kartu = st.columns([1, 1.5], gap='medium')
        with kolom_donat:
            st.markdown(tampilan.donat_html(hasil['persen_tergenang'], hasil['status']), unsafe_allow_html=True)
        with kolom_kartu:
            ambang = meta['status']
            st.markdown(tampilan.kartu_html([
                ('luas', 'Luas tergenang', f'{hasil["luas_tergenang_km2"]:,.2f} km²',
                 f'{hasil["luas_tergenang_km2"] * 100:,.0f} ha'),
                ('perubahan', 'Perubahan', f'{hasil["perubahan_km2"]:+,.2f} km²',
                 f'dibanding {inti.format_tanggal(ringkas_awal["tanggal"])} ({ringkas_awal["luas_air_km2"]:,.2f} km²)'),
                ('persen', 'Persen daratan tergenang', f'{hasil["persen_tergenang"]:.2f}%',
                 f'{hasil["n_tergenang"]:,} dari {hasil["n_darat"]:,} piksel darat'),
                ('ambang', 'Ambang status', hasil['status'],
                 f'Aman &lt; {ambang["waspada"]:.2f}% ≤ Waspada &lt; {ambang["bahaya"]:.2f}% ≤ Bahaya'),
            ]), unsafe_allow_html=True)

        nama_berkas = 'prediksi_' + inti.tanggal_prediksi(meta, sk['gap_days']).strftime('%Y%m%d')
        kolom_unduh1, kolom_unduh2 = st.columns(2)
        with kolom_unduh1:
            st.download_button(
                'Unduh peta PNG',
                data=tampilan.png_peta(rgb, f'{mode} - prediksi {tgl_hasil}',
                                       f'{meta["nama_model"]} | tergenang {hasil["persen_tergenang"]:.2f}% '
                                       f'({hasil["luas_tergenang_km2"]:.2f} km2) | status {hasil["status"]}',
                                       [(w, l.replace('²', '2')) for w, l in legenda]),
                file_name=f'{nama_berkas}_{mode.split()[0].lower()}.png', mime='image/png', width='stretch',
            )
        with kolom_unduh2:
            st.download_button(
                'Unduh ringkasan CSV',
                data=inti.tabel_ringkasan(meta, sk, hasil).to_csv(index=False).encode('utf-8-sig'),
                file_name=f'{nama_berkas}_ringkasan.csv', mime='text/csv', width='stretch',
            )

with st.expander('Tentang model'):
    st.markdown(f'**{meta["nama_model"]}** ({meta["jenis_model"]}), diekspor {meta["tanggal_ekspor"]}, torch {meta["versi_torch"]}.')
    st.markdown('**Perbandingan model pada data test**')
    df_banding = pd.DataFrame(meta['pembanding']).rename(columns={
        'nama': 'Model', 'jenis': 'Jenis', 'PR_AUC_dyn': 'PR-AUC dinamis', 'F1': 'F1', 'Infer_s': 'Inferensi test (s)'})
    df_banding['Dideploy'] = df_banding['Model'].eq(meta['nama_model']).map({True: 'Ya', False: ''})
    st.dataframe(df_banding.round(4), hide_index=True, width='stretch')
    mt = meta['metrik_test']
    st.markdown(f'**Metrik test {meta["nama_model"]}**')
    st.dataframe(pd.DataFrame([
        ('PR-AUC piksel dinamis (metrik utama)', f'{mt["PR_AUC_dyn"]:.4f}', mt['PR_AUC_dyn_CI']),
        ('PR-AUC semua piksel', f'{mt["PR_AUC"]:.4f}', ''),
        ('F1', f'{mt["F1"]:.4f}', ''),
        ('Recall', f'{mt["Recall"]:.4f}', ''),
        ('Precision', f'{mt["Precision"]:.4f}', ''),
        ('Recall genangan baru', f'{mt["Recall_new"]:.4f}', ''),
        ('MAE luas per sel grid (km²)', f'{mt["MAE_km2_sel"]:.4f}', ''),
        ('R² luas per sel grid', f'{mt["R2_sel"]:.4f}', ''),
    ], columns=['Metrik', 'Nilai', 'Nilai dan CI 95%']), hide_index=True, width='stretch')
    if meta['alasan_rekomendasi']:
        st.markdown('**Alasan rekomendasi**')
        st.markdown('\n'.join(f'- {l.lstrip("- ")}' for l in meta['alasan_rekomendasi']))
    st.markdown(
        f'**Input model**: {meta["seq_len"]} frame ({meta["seq_len"] - 1} observasi riwayat dan 1 frame skenario), '
        f'{len(meta["frame_ch"])} kanal: {", ".join(meta["frame_ch"])}. '
        f'Threshold tergenang {meta["threshold"]:.4f}, tile {meta["ukuran_tile"]} piksel dengan overlap {meta["overlap_tile"]}.'
    )
    st.markdown(
        '**Keterbatasan**\n'
        '- Slider hanya mengubah intensitas (rata-rata darat) hujan; pola spasial hujan tetap mengikuti observasi terakhir.\n'
        '- Data MODIS (NDVI, NDWI) frame target memakai scene terakhir, tidak disimulasikan.\n'
        '- Label air berasal dari ambang backscatter VV Sentinel-1 sebesar −15 dB, sehingga bisa tertukar dengan '
        'permukaan halus lain atau terlewat di area bervegetasi dan perkotaan.\n'
        '- Status Aman, Waspada, dan Bahaya ditentukan dari persentil 75 dan 95 persentase genangan historis, '
        'bukan standar resmi kebencanaan.\n'
        '- Model belajar terutama dari riwayat genangan dan frekuensi historis, sehingga respons terhadap perubahan '
        'hujan bisa terbatas. Selisih antar preset kecil adalah perilaku model, bukan kesalahan aplikasi.'
    )