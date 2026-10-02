# topik1/app.py
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
IKON_MULAI = ':material/hourglass_top:'
IKON_SUKSES = ':material/check_circle:'
IKON_GAGAL = ':material/error:'
IKON_INFO = ':material/info:'

st.set_page_config(page_title='Water on Jabodetabek', layout='wide')
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
TGL_AKHIR = inti.tanggal_riwayat_akhir(meta)
MODIS_SIAP = inti.norm_tersedia(meta, 'ndvi') and inti.norm_tersedia(meta, 'ndwi')


@st.cache_data(max_entries=8, show_spinner=False)
def prediksi_tersimpan(kunci):
    return inti.untuk_cache(inti.prediksi_skenario(ambil_artefak(), dict(kunci)))


@st.cache_resource(show_spinner=False)
def verifikasi_awal():
    verify = inti.muat_verify(FOLDER_ARTEFAK)
    hasil = inti.prediksi_skenario(ambil_artefak(), verify['skenario'])
    return inti.cek_verifikasi(verify, hasil)


def rahasia(kunci):
    try:
        return str(st.secrets.get(kunci, ''))
    except Exception:
        return ''


def hari_ini():
    return datetime.now(ZONA_WIB).date()


def nilai_slider(kunci, nilai):
    s = meta['slider'][kunci]
    v = min(max(float(nilai), float(s['min'])), float(s['maks']))
    return int(round(v)) if isinstance(s['langkah'], int) else float(v)


def terapkan_preset(nama):
    for k, v in meta['preset'][nama].items():
        if k in meta['slider']:
            st.session_state[f'sl_{k}'] = nilai_slider(k, v)
    st.session_state['preset_aktif'] = nama


def set_hari_ini():
    st.session_state['tgl_cek'] = hari_ini()


def format_angka(v, desimal=1):
    return f'{v:,.{desimal}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


for k, s in meta['slider'].items():
    st.session_state.setdefault(f'sl_{k}', nilai_slider(k, s['default']))
st.session_state.setdefault('tgl_cek', max(hari_ini(), TGL_AKHIR + timedelta(days=1)))
st.session_state.setdefault('nasa_token', rahasia('NASA_EARTHDATA_TOKEN'))
st.session_state.setdefault('nasa_user', rahasia('NASA_EARTHDATA_USER'))
st.session_state.setdefault('nasa_pass', rahasia('NASA_EARTHDATA_PASSWORD'))
st.session_state.setdefault('cache_tanggal', {})

teks_riwayat = (f'Riwayat: observasi Sentinel-1 {", ".join(inti.format_tanggal(t) for t in meta["tanggal_riwayat"])} '
                f'· luas air terakhir {format_angka(meta["kondisi_awal_ringkas"]["luas_air_km2"])} km²')
st.markdown(tampilan.kepala_html(teks_riwayat), unsafe_allow_html=True)

with st.spinner(f'Menyiapkan dan memverifikasi model, sekitar {DETIK:.0f} detik pada kunjungan pertama...'):
    hasil_cek = verifikasi_awal()
if not hasil_cek['lulus']:
    st.warning(
        'Verifikasi awal tidak lolos: hasil aplikasi berbeda dari acuan notebook '
        f'(selisih persen {hasil_cek["selisih_persen"]:.4f} poin, toleransi {hasil_cek["toleransi_persen"]}; '
        f'selisih probabilitas maks {hasil_cek["selisih_prob_maks"]:.5f}, toleransi {hasil_cek["toleransi_prob"]}). '
        'Periksa versi torch dan kelengkapan artefak. Aplikasi tetap berjalan.'
    )
    if not st.session_state.get('toast_verifikasi'):
        st.toast('Verifikasi awal model tidak lolos. Lihat peringatan di atas.', icon=IKON_GAGAL)
        st.session_state['toast_verifikasi'] = True


def simpan_hasil(entri):
    st.session_state['hasil'] = entri


def jalankan_skenario(skenario, kunci):
    st.toast('Prediksi skenario dimulai.', icon=IKON_MULAI)
    with st.status(f'Menjalankan model, perkiraan sekitar {DETIK:.0f} detik...', expanded=False) as status:
        t0 = time.perf_counter()
        try:
            hasil = prediksi_tersimpan(kunci)
        except Exception as e:
            status.update(label='Prediksi gagal', state='error', expanded=True)
            st.error(f'Prediksi gagal: {e}')
            st.toast('Prediksi gagal. Lihat pesan galat.', icon=IKON_GAGAL)
            return
        detik = time.perf_counter() - t0
        status.update(label=f'Prediksi selesai dalam {detik:.1f} detik', state='complete')
    simpan_hasil({'sumber': 'Simulasi skenario', 'kunci': kunci, 'tanggal': inti.tanggal_prediksi(meta, skenario['gap_days']),
                  'skenario': dict(skenario), 'hasil': hasil, 'detik': detik, 'catatan': [], 'data': []})
    st.toast(f'Prediksi berhasil: {hasil["persen_tergenang"]:.2f}% daratan tergenang ({hasil["status"]}).', icon=IKON_SUKSES)


def ambil_data_tanggal(tgl, pakai_modis, kred, lapor):
    kunci = (tgl.isoformat(), pakai_modis)
    simpanan = st.session_state['cache_tanggal']
    if kunci in simpanan:
        lapor('Memakai data yang sudah diunduh sebelumnya')
        return simpanan[kunci]
    lon_px, lat_px = inti.koordinat_piksel(meta)
    extent = meta['grid']['extent']
    hujan = data_nasa.ambil_hujan(kred, tgl, extent, lon_px, lat_px, lapor=lapor)
    data = {k: hujan[k].astype(np.float16) for k in inti.KANAL_HUJAN}
    catatan = list(hujan['catatan'])
    info = [f'GPM IMERG {"/".join(hujan["produk"])}, jendela hujan berakhir {inti.format_tanggal(hujan["akhir_jendela"])}']
    if pakai_modis:
        try:
            modis = data_nasa.ambil_modis(tgl, extent, lon_px, lat_px, lapor=lapor)
            data['ndvi'] = modis['ndvi'].astype(np.float16)
            data['ndwi'] = modis['ndwi'].astype(np.float16)
            cakupan = float(np.isfinite(modis['ndwi'][darat]).mean() * 100)
            info.append(f'MODIS {data_nasa.PRODUK_MODIS} komposit {inti.format_tanggal(modis["komposit"])} '
                        f'(cakupan bebas awan {cakupan:.0f}%)')
        except data_nasa.GagalUnduh as e:
            catatan.append(f'MODIS tidak dapat diambil ({e}); memakai MODIS scene terakhir.')
    else:
        info.append(f'MODIS memakai scene terakhir ({inti.format_tanggal(TGL_AKHIR)})')
    hasil = {'peta': data, 'catatan': catatan, 'info': info}
    if len(simpanan) >= 2:
        simpanan.pop(next(iter(simpanan)))
    simpanan[kunci] = hasil
    return hasil


def jalankan_tanggal(tgl, pakai_modis, kred):
    st.toast(f'Mengambil data NASA untuk {inti.format_tanggal(tgl)}...', icon=IKON_MULAI)
    with st.status('Menyiapkan data satelit...', expanded=True) as status:
        t0 = time.perf_counter()
        try:
            data = ambil_data_tanggal(tgl, pakai_modis, kred, lambda teks: status.update(label=teks))
            status.update(label=f'Menjalankan model, perkiraan sekitar {DETIK:.0f} detik...')
            bar = st.progress(0.0, text='Inferensi bertile')
            terakhir = {'v': -1}

            def lapor_tile(f):
                langkah = int(f * 20)
                if langkah != terakhir['v']:
                    terakhir['v'] = langkah
                    bar.progress(min(f, 1.0), text=f'Inferensi bertile {f * 100:.0f}%')

            gap_nyata, gap_model = inti.gap_untuk_tanggal(meta, tgl)
            peta = {k: v.astype(np.float32) for k, v in data['peta'].items()}
            skenario = {x: inti.rata_darat(darat, peta.get(x)) for x in inti.KANAL_HUJAN}
            skenario['gap_days'] = gap_model
            hasil = inti.untuk_cache(inti.prediksi_skenario(artefak, skenario, peta_input=peta, lapor=lapor_tile))
            del peta
        except data_nasa.GagalUnduh as e:
            status.update(label='Pengambilan data gagal', state='error', expanded=True)
            st.error(str(e))
            st.toast('Gagal mengambil data NASA. Lihat pesan galat.', icon=IKON_GAGAL)
            return
        except Exception as e:
            status.update(label='Prediksi gagal', state='error', expanded=True)
            st.error(f'Prediksi gagal: {e}')
            st.toast('Prediksi gagal. Lihat pesan galat.', icon=IKON_GAGAL)
            return
        detik = time.perf_counter() - t0
        status.update(label=f'Selesai dalam {detik:.1f} detik', state='complete', expanded=False)
    catatan = [f'Model memakai riwayat Sentinel-1 terakhir ({inti.format_tanggal(TGL_AKHIR)}). Kondisi permukaan '
               'setelah tanggal itu tidak diketahui model, sehingga hasil adalah estimasi berbasis hujan aktual.']
    if gap_nyata != gap_model:
        catatan.append(f'Jarak nyata {gap_nyata} hari di luar rentang latih '
                       f'({meta["slider"]["gap_days"]["min"]}–{meta["slider"]["gap_days"]["maks"]} hari); '
                       f'model diberi {gap_model} hari.')
    catatan += data['catatan']
    simpan_hasil({'sumber': 'Cek tanggal', 'kunci': ('tanggal', tgl.isoformat(), pakai_modis), 'tanggal': tgl,
                  'skenario': skenario, 'hasil': hasil, 'detik': detik, 'catatan': catatan, 'data': data['info']})
    st.toast(f'Prediksi {inti.format_tanggal(tgl)} berhasil: {hasil["persen_tergenang"]:.2f}% tergenang '
             f'({hasil["status"]}).', icon=IKON_SUKSES)


tab_skenario, tab_tanggal = st.tabs(['Simulasi skenario', 'Cek tanggal'])

with tab_skenario:
    with st.container(border=True):
        st.markdown(tampilan.label_kecil('Preset skenario'), unsafe_allow_html=True)
        kolom_preset = st.columns(len(meta['preset']))
        for kol, nama in zip(kolom_preset, meta['preset']):
            kol.button(nama, on_click=terapkan_preset, args=(nama,), width='stretch', key=f'preset_{nama}',
                       type='secondary')
        st.write('')
        kolom_slider = st.columns(2, gap='large')
        for i, (k, s) in enumerate(meta['slider'].items()):
            with kolom_slider[i % 2]:
                konversi = int if isinstance(s['langkah'], int) else float
                st.slider(f'{s["label"]} ({s["satuan"]})', min_value=konversi(s['min']),
                          max_value=konversi(s['maks']), step=konversi(s['langkah']), key=f'sl_{k}')
                st.markdown(f'<div class="rentang">Rentang historis {s["min"]}–{s["maks"]} {s["satuan"]}</div>',
                            unsafe_allow_html=True)
        skenario_ui = {k: st.session_state[f'sl_{k}'] for k in URUTAN}
        kunci_ui = inti.kunci_skenario(skenario_ui, URUTAN)
        tgl_skenario = inti.tanggal_prediksi(meta, skenario_ui['gap_days'])
        st.markdown(f'<div class="info-tanggal">{tampilan.ikon_ui("kalender")}Tanggal prediksi '
                    f'<b>{inti.format_tanggal(tgl_skenario)}</b> ({skenario_ui["gap_days"]} hari setelah observasi terakhir)</div>',
                    unsafe_allow_html=True)
        _, kolom_tombol, _ = st.columns([1, 1.2, 1])
        tekan_skenario = kolom_tombol.button('Jalankan prediksi', type='primary', width='stretch', key='tombol_skenario',
                                             icon=':material/play_arrow:')
        terakhir = st.session_state.get('hasil')
        if terakhir and terakhir['sumber'] == 'Simulasi skenario' and terakhir['kunci'] != kunci_ui:
            st.markdown(tampilan.catatan_html('Skenario berubah sejak prediksi terakhir. Tekan Jalankan prediksi '
                                              'untuk memperbarui hasil.'), unsafe_allow_html=True)
    if tekan_skenario:
        jalankan_skenario(skenario_ui, kunci_ui)

with tab_tanggal:
    with st.container(border=True):
        st.markdown(
            '<div class="konteks">Prediksi memakai curah hujan GPM IMERG aktual (24 jam, 72 jam, dan 7 hari sampai '
            'tanggal yang dipilih) dan, bila diaktifkan, komposit MODIS terbaru. Riwayat Sentinel-1 tetap observasi '
            f'terakhir ({inti.format_tanggal(TGL_AKHIR)}).</div>', unsafe_allow_html=True)
        kred_lengkap = data_nasa.kredensial_lengkap({'token': st.session_state['nasa_token'],
                                                     'user': st.session_state['nasa_user'],
                                                     'password': st.session_state['nasa_pass']})
        with st.expander('Akses NASA Earthdata' + ('' if kred_lengkap else ' (wajib diisi)'), expanded=not kred_lengkap,
                         icon=':material/key:'):
            st.caption('Isi salah satu: token Earthdata, atau username dan password. Kredensial hanya disimpan di sesi '
                       'browser ini dan tidak ditulis ke disk. Token bisa dibuat di urs.earthdata.nasa.gov, menu '
                       'Generate Token.')
            st.text_input('NASA_EARTHDATA_TOKEN', type='password', key='nasa_token')
            kolom_u, kolom_p = st.columns(2)
            kolom_u.text_input('NASA_EARTHDATA_USER', key='nasa_user')
            kolom_p.text_input('NASA_EARTHDATA_PASSWORD', type='password', key='nasa_pass')
        kolom_tgl, kolom_hari = st.columns([2, 1], vertical_alignment='bottom')
        kolom_tgl.date_input('Tanggal prediksi', min_value=TGL_AKHIR + timedelta(days=1),
                             max_value=max(hari_ini(), TGL_AKHIR + timedelta(days=1)), key='tgl_cek',
                             format='DD/MM/YYYY')
        kolom_hari.button('Cek hari ini', on_click=set_hari_ini, width='stretch', icon=':material/today:')
        tgl_cek = st.session_state['tgl_cek']
        gap_nyata, gap_model = inti.gap_untuk_tanggal(meta, tgl_cek)
        st.markdown(f'<div class="info-tanggal">{tampilan.ikon_ui("kalender")}{gap_nyata} hari setelah observasi '
                    f'Sentinel-1 terakhir</div>', unsafe_allow_html=True)
        if gap_nyata != gap_model:
            st.markdown(tampilan.catatan_html(
                f'Jarak {gap_nyata} hari melebihi rentang data latih (maks {meta["slider"]["gap_days"]["maks"]} hari). '
                f'Model akan diberi {gap_model} hari, dan hasil perlu dibaca sebagai estimasi kasar.'), unsafe_allow_html=True)
        pakai_modis = st.checkbox('Gunakan MODIS terbaru (layanan subset ORNL, tanpa login)', value=MODIS_SIAP,
                                  disabled=not MODIS_SIAP, key='pakai_modis')
        if not MODIS_SIAP:
            st.caption('MODIS terbaru belum bisa dipakai karena meta.json belum memuat norm_modis. Model memakai '
                       'MODIS scene terakhir.')
        _, kolom_tombol2, _ = st.columns([1, 1.2, 1])
        tekan_tanggal = kolom_tombol2.button('Cek tanggal ini', type='primary', width='stretch', key='tombol_tanggal',
                                             icon=':material/satellite_alt:', disabled=not kred_lengkap)
        if not kred_lengkap:
            st.caption('Isi kredensial NASA Earthdata untuk mengaktifkan tombol.')
        terakhir = st.session_state.get('hasil')
        if terakhir and terakhir['sumber'] == 'Cek tanggal' and terakhir['kunci'] != ('tanggal', tgl_cek.isoformat(), pakai_modis):
            st.markdown(tampilan.catatan_html('Tanggal atau pilihan data berubah sejak prediksi terakhir. Tekan Cek '
                                              'tanggal ini untuk memperbarui.'), unsafe_allow_html=True)
    if tekan_tanggal:
        jalankan_tanggal(tgl_cek, pakai_modis, {'token': st.session_state['nasa_token'],
                                                'user': st.session_state['nasa_user'],
                                                'password': st.session_state['nasa_pass']})

with st.expander('Pengaturan lanjutan', icon=':material/tune:'):
    pakai_thr_validasi = st.toggle(f'Gunakan threshold validasi model ({meta["threshold"]:.4f})', value=True,
                                   key='thr_validasi')
    if pakai_thr_validasi:
        threshold = float(meta['threshold'])
    else:
        threshold = st.slider('Threshold keputusan', min_value=0.05, max_value=0.95, step=0.01,
                              value=round(float(meta['threshold']), 2), key='thr_manual')
        st.caption('Threshold lebih rendah membuat lebih banyak piksel dianggap tergenang. Angka evaluasi di '
                   'skripsi memakai threshold validasi.')


def tampilkan_hasil(entri):
    hasil = inti.terapkan_threshold(meta, darat, entri['hasil'], threshold)
    sk = entri['skenario']
    st.markdown(tampilan.label_kecil(f'Hasil · {entri["sumber"]}'), unsafe_allow_html=True)
    st.markdown(f'<h3 style="text-align:center;margin:0 0 .3rem 0">Prediksi {inti.format_tanggal(entri["tanggal"])}</h3>',
                unsafe_allow_html=True)
    for c in entri['catatan']:
        st.markdown(tampilan.catatan_html(c), unsafe_allow_html=True)
    perubahan = hasil['perubahan_km2']
    st.markdown(tampilan.statistik_html([
        ('Luas genangan prediksi', f'{format_angka(hasil["luas_tergenang_km2"])} km²',
         f'{format_angka(hasil["luas_tergenang_km2"] * 100, 0)} ha'),
        ('Perubahan dari scene terakhir', f'{"+" if perubahan >= 0 else "−"}{format_angka(abs(perubahan))} km²',
         f'dibanding {inti.format_tanggal(meta["kondisi_awal_ringkas"]["tanggal"])}'),
        ('Persen daratan tergenang', f'{format_angka(hasil["persen_tergenang"], 2)} %',
         f'{hasil["n_tergenang"]:,} dari {hasil["n_darat"]:,} piksel darat'.replace(',', '.')),
    ]), unsafe_allow_html=True)

    kolom_donat, kolom_konteks = st.columns([1, 1.7], gap='large', vertical_alignment='center')
    with kolom_donat:
        st.markdown(tampilan.donat_html(hasil['persen_tergenang'], hasil['status']), unsafe_allow_html=True)
    with kolom_konteks:
        hujan = ' · '.join(f'{meta["slider"][x]["label"].replace("Curah hujan ", "")} '
                           f'<b>{format_angka(sk[x])} mm</b>' for x in inti.KANAL_HUJAN if sk.get(x) is not None)
        ambang = meta['status']
        baris = [f'Curah hujan rata-rata darat: {hujan}',
                 f'Jarak ke observasi berikutnya: <b>{sk["gap_days"]} hari</b>',
                 f'Status: <b>Aman</b> &lt; {format_angka(ambang["waspada"], 2)}% ≤ <b>Waspada</b> &lt; '
                 f'{format_angka(ambang["bahaya"], 2)}% ≤ <b>Bahaya</b> (persentil 75 dan 95 historis)',
                 f'Threshold keputusan: <b>{hasil["threshold"]:.4f}</b>'
                 + (' (validasi)' if abs(hasil['threshold'] - meta['threshold']) < 1e-9 else ' (manual)')]
        baris += [f'Data: {d}' for d in entri['data']]
        baris.append(f'Model {meta["nama_model"]} · waktu proses {entri["detik"]:.1f} detik')
        st.markdown(tampilan.konteks_html(baris), unsafe_allow_html=True)

    H, W = darat.shape
    f = inti.faktor_tampilan(H, W)
    prob_tampil = inti.rerata_blok(hasil['prob'], darat, f)
    lon, lat, rasio = inti.koordinat_tampilan(meta, *prob_tampil.shape, f)
    legenda_prob = [(tampilan.SKALA_PROB[0][1], 'rendah'), (tampilan.NAVY, 'tinggi'), (tampilan.LAUT, 'laut')]
    _, kolom_mode = st.columns([1, 1])
    mode = kolom_mode.radio('Peta kanan', ['Peta biner', 'Perubahan'], horizontal=True, key='mode_kanan',
                            label_visibility='collapsed')
    kolom_kiri, kolom_kanan = st.columns(2, gap='medium')
    with kolom_kiri:
        st.markdown(tampilan.label_kecil('Probabilitas genangan'), unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(tampilan.figur_probabilitas(prob_tampil, lon, lat, rasio), width='stretch',
                            config={'displaylogo': False, 'displayModeBar': False}, key='peta_prob')
        st.markdown(tampilan.legenda_html(legenda_prob), unsafe_allow_html=True)
    with kolom_kanan:
        if mode == 'Perubahan':
            kelas_penuh = inti.kelas_perubahan(hasil['air'], artefak['kondisi']['air_terakhir'], darat)
            kelas = inti.kelas_mayoritas(kelas_penuh, f, len(inti.KELAS_PERUBAHAN), 4)
            luas = inti.luas_perubahan(meta, kelas_penuh)
            label = [f'{n} ({format_angka(luas[n])} km²)' if n in luas else n for n in inti.KELAS_PERUBAHAN]
            warna = tampilan.WARNA_PERUBAHAN
            judul_kanan = f'Perubahan dibanding {inti.format_tanggal(meta["kondisi_awal_ringkas"]["tanggal"])}'
        else:
            kelas = inti.kelas_mayoritas(inti.kelas_banjir(hasil['air'], darat), f, len(inti.KELAS_BANJIR), 2)
            label = ['kering', 'air', 'laut']
            warna = tampilan.WARNA_BANJIR
            judul_kanan = f'Peta biner (threshold {hasil["threshold"]:.2f})'
        st.markdown(tampilan.label_kecil(judul_kanan), unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(tampilan.figur_kelas(kelas, lon, lat, rasio, warna, label), width='stretch',
                            config={'displaylogo': False, 'displayModeBar': False}, key='peta_kanan')
        legenda_kanan = list(zip(warna, label))
        st.markdown(tampilan.legenda_html(legenda_kanan), unsafe_allow_html=True)
    if f > 1:
        st.caption(f'Peta ditampilkan per blok {f}×{f} piksel. Semua angka dihitung pada resolusi penuh.')

    nama_berkas = 'water_on_jabodetabek_' + inti.ke_tanggal(entri['tanggal']).strftime('%Y%m%d')
    png = tampilan.png_laporan(
        [(tampilan.rgb_probabilitas(prob_tampil), 'Probabilitas genangan', legenda_prob),
         (tampilan.rgb_kelas(kelas, warna), judul_kanan, [(w, l.replace('²', '2')) for w, l in legenda_kanan])],
        f'Water on Jabodetabek - prediksi {inti.format_tanggal(entri["tanggal"])}',
        f'{entri["sumber"]} | {meta["nama_model"]} | tergenang {hasil["persen_tergenang"]:.2f}% '
        f'({hasil["luas_tergenang_km2"]:.1f} km2) | status {hasil["status"]} | threshold {hasil["threshold"]:.3f}')
    kolom_u1, kolom_u2 = st.columns(2)
    kolom_u1.download_button('Unduh peta PNG', data=png, file_name=f'{nama_berkas}.png', mime='image/png',
                             width='stretch', icon=':material/image:')
    kolom_u2.download_button('Unduh ringkasan CSV',
                             data=inti.tabel_ringkasan(meta, entri, hasil).to_csv(index=False).encode('utf-8-sig'),
                             file_name=f'{nama_berkas}.csv', mime='text/csv', width='stretch',
                             icon=':material/table_view:')


st.write('')
entri = st.session_state.get('hasil')
if entri:
    tampilkan_hasil(entri)
else:
    st.markdown(f'<div class="kosong">{tampilan.ikon_ui("info")} Atur skenario lalu tekan <b>Jalankan prediksi</b>, '
                'atau buka tab <b>Cek tanggal</b>.</div>', unsafe_allow_html=True)

st.write('')
with st.expander('Tentang model', icon=':material/info:'):
    st.markdown(f'**{meta["nama_model"]}** ({meta["jenis_model"]}), diekspor {meta["tanggal_ekspor"]}, '
                f'torch {meta["versi_torch"]}.')
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
        f'**Input model**: {meta["seq_len"]} frame ({meta["seq_len"] - 1} observasi riwayat dan 1 frame target), '
        f'{len(meta["frame_ch"])} kanal: {", ".join(meta["frame_ch"])}. Threshold validasi {meta["threshold"]:.4f}, '
        f'tile {meta["ukuran_tile"]} piksel dengan overlap {meta["overlap_tile"]}.'
    )
    st.markdown(
        '**Keterbatasan**\n'
        '- Pada simulasi skenario, slider hanya mengubah intensitas hujan; pola spasial hujan mengikuti observasi terakhir.\n'
        '- Pada simulasi skenario, MODIS frame target memakai scene terakhir.\n'
        '- Cek tanggal memakai hujan GPM aktual, tetapi riwayat Sentinel-1 tetap observasi terakhir. Semakin jauh '
        'tanggal dari observasi itu, semakin kasar estimasinya. GPM diutamakan versi Final, lalu Late dan Early bila '
        'Final belum terbit; MODIS memakai komposit 8 harian MOD09A1, bukan citra harian.\n'
        '- Label air berasal dari ambang backscatter VV Sentinel-1 sebesar −15 dB.\n'
        '- Status Aman, Waspada, dan Bahaya berasal dari persentil 75 dan 95 persentase genangan historis, '
        'bukan standar resmi kebencanaan.\n'
        '- Model belajar terutama dari riwayat genangan dan frekuensi historis, sehingga respons terhadap hujan '
        'terbatas. Selisih antar skenario yang kecil adalah perilaku model, bukan kesalahan aplikasi.'
    )