# tampilan.py
import numpy as np
import plotly.graph_objects as go

NAVY = '#1f2a44'
GARIS = '#e6e3dc'
REDUP = '#5f636c'
KERING = '#ece9e2'
LAUT = '#d5dce5'
WARNA_ANOMALI = [KERING, '#9db3d1', NAVY, '#d9a441', LAUT]
LABEL_ANOMALI = ['Kering', 'Tergenang seperti biasa', 'Lebih basah dari biasa', 'Lebih kering dari biasa', 'Laut']
URUTAN_LEGENDA = [2, 1, 3, 0, 4]
WARNA_KATEGORI = {'Di bawah normal': '#3f7d58', 'Normal': '#4d5d78', 'Di atas normal': '#b23b3b'}
IKON_KATEGORI = {'Di bawah normal': 'M12 6.5v10M7.5 12.5l4.5 4.5 4.5-4.5',
                 'Normal': 'M7.5 9.8h9M7.5 14.2h9',
                 'Di atas normal': 'M12 17.5v-10M7.5 11.5l4.5-4.5 4.5 4.5'}
WARNA_YAKIN = {'Data asli': '#3f7d58', 'Tinggi': '#3f7d58', 'Sedang': '#9a6a12', 'Rendah': '#b23b3b',
               'Belum teruji': REDUP}

CSS = (
    '<style>'
    "@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');"
    'html,body,[class*="css"],.stMarkdown,.stButton,.stDateInput,.stExpander{font-family:"Nunito Sans",system-ui,sans-serif}'
    '.block-container{max-width:880px;padding-top:4.5rem;padding-bottom:4rem}'
    '[data-testid="stVerticalBlockBorderWrapper"]{background:#ffffff;border-color:#e6e3dc !important;border-radius:10px}'
    '.stButton>button[kind="primary"]{background:#1f2a44;border-color:#1f2a44;min-height:3rem;font-weight:700;letter-spacing:.02em}'
    '.stButton>button[kind="primary"]:hover{background:#2c3a5c;border-color:#2c3a5c}'
    '.stDateInput input{font-size:1.1rem;min-height:2.6rem}'
    '[data-testid="stExpander"] details{border:none;border-top:1px solid #e6e3dc;border-radius:0;background:transparent}'
    '[data-testid="stExpander"] summary{font-weight:700;font-size:1.05rem;color:#1f2a44;padding:1rem .2rem}'
    '.alis{font-size:.75rem;letter-spacing:.2em;text-transform:uppercase;color:#5f636c}'
    '.judul{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:2.75rem;line-height:1.1;color:#1f2a44;margin:.3rem 0 .5rem}'
    '.pengantar{font-size:1.06rem;line-height:1.55;color:#4b5160;max-width:620px;margin-bottom:1.4rem}'
    '.tanya{font-family:"Source Serif 4",Georgia,serif;font-size:1.4rem;font-weight:600;color:#1f2a44;margin-bottom:.2rem}'
    '.kalimat{display:flex;gap:.9rem;align-items:flex-start;margin:1.6rem 0 1.2rem}'
    '.kalimat p{margin:0;font-family:"Source Serif 4",Georgia,serif;font-size:1.6rem;line-height:1.35;color:#1f2a44}'
    '.statistik{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));border-top:1px solid #e6e3dc;border-bottom:1px solid #e6e3dc;margin-bottom:1.4rem}'
    '.stat{padding:1.3rem 1rem;text-align:center;border-left:1px solid #e6e3dc}'
    '.stat:first-child{border-left:none}'
    '.stat-label{font-size:.75rem;letter-spacing:.16em;text-transform:uppercase;color:#5f636c}'
    '.stat-nilai{font-family:"Source Serif 4",Georgia,serif;font-size:2.2rem;color:#1f2a44;margin-top:.35rem;line-height:1.15}'
    '.stat-ket{font-size:.82rem;color:#5f636c;margin-top:.25rem}'
    '.label-peta{font-size:.75rem;letter-spacing:.16em;text-transform:uppercase;color:#5f636c;margin-bottom:.4rem}'
    '.legenda{display:flex;flex-wrap:wrap;justify-content:center;gap:.4rem 1.1rem;font-size:.88rem;color:#4b5160;margin:.6rem 0}'
    '.kotak{display:inline-block;width:.85rem;height:.85rem;border-radius:3px;border:1px solid #cfcac0;margin-right:.4rem;vertical-align:-.1rem}'
    '.ket-peta{font-size:.82rem;color:#5f636c;text-align:center;line-height:1.6;margin-bottom:1.4rem}'
    '.catatan{background:#fbf6ea;border:1px solid #efe1bd;color:#6b4f0f;border-radius:6px;padding:.6rem .8rem;font-size:.88rem;margin-top:.4rem}'
    '.isi-lipat{color:#4b5160;line-height:1.65}'
    '.isi-lipat h4{font-family:"Source Serif 4",Georgia,serif;font-size:1.12rem;font-weight:600;color:#1f2a44;margin:1.2rem 0 .3rem}'
    '.isi-lipat h4:first-child{margin-top:.2rem}'
    '.isi-lipat ul,.isi-lipat ol{margin:0;padding-left:1.3rem}'
    '.isi-lipat li{margin-bottom:.35rem}'
    '.batang{display:flex;align-items:center;gap:.7rem;font-size:.9rem;margin:.35rem 0}'
    '.batang .lb{flex:0 0 150px;color:#4b5160}'
    '.batang .bg{flex:1 1 auto;height:14px;background:#efede7;border-radius:3px;overflow:hidden}'
    '.batang .isi{display:block;height:100%;background:#1f2a44}'
    '.batang .sk{flex:0 0 44px;text-align:right;font-weight:700;color:#1f2a44}'
    '@media (max-width:640px){.judul{font-size:2.1rem}.kalimat p{font-size:1.3rem}.stat{border-left:none;border-top:1px solid #e6e3dc}.stat:first-child{border-top:none}.batang .lb{flex-basis:110px}}'
    '</style>'
)


def kepala_html():
    return ('<div class="alis">Prediksi genangan · Jabodetabek</div>'
            '<div class="judul">Water on Jabodetabek</div>'
            '<div class="pengantar">Pilih tanggal, lalu lihat daerah mana yang diperkirakan tergenang dan apakah lebih '
            'parah dari biasanya.</div>')


def kalimat_html(tanggal, kata, kategori, bulan):
    w = WARNA_KATEGORI[kategori]
    return ('<div class="kalimat">'
            f'<svg width="40" height="40" viewBox="0 0 24 24" aria-hidden="true" style="flex:none;margin-top:2px">'
            f'<circle cx="12" cy="12" r="11" fill="{w}"/><path d="{IKON_KATEGORI[kategori]}" fill="none" stroke="#fff" '
            'stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            f'<p>Pada <strong>{tanggal}</strong>, genangan di Jabodetabek {kata} '
            f'<strong style="color:{w}">{kategori.upper()}</strong> untuk bulan {bulan}.</p></div>')


def statistik_html(daftar):
    isi = ''.join(f'<div class="stat"><div class="stat-label">{l}</div>'
                  f'<div class="stat-nilai" style="color:{w}">{n}</div><div class="stat-ket">{k}</div></div>'
                  for l, n, k, w in daftar)
    return f'<div class="statistik">{isi}</div>'


def legenda_html():
    return '<div class="legenda">' + ''.join(
        f'<span><span class="kotak" style="background:{WARNA_ANOMALI[i]}"></span>{LABEL_ANOMALI[i]}</span>'
        for i in URUTAN_LEGENDA) + '</div>'


def catatan_html(teks):
    return f'<div class="catatan">{teks}</div>'


def figur_peta(kelas, lon, lat, rasio):
    n = len(WARNA_ANOMALI)
    skala = []
    for i, w in enumerate(WARNA_ANOMALI):
        skala += [[i / n, w], [(i + 1) / n, w]]
    teks = np.asarray(LABEL_ANOMALI, dtype=object)[kelas]
    fig = go.Figure(go.Heatmap(z=kelas, x=lon, y=lat, colorscale=skala, zmin=-0.5, zmax=n - 0.5, showscale=False,
                               customdata=teks, hovertemplate='%{customdata}<extra></extra>'))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                      xaxis=dict(visible=False, constrain='domain'),
                      yaxis=dict(visible=False, scaleanchor='x', scaleratio=rasio, constrain='domain'),
                      hoverlabel=dict(bgcolor='#ffffff', font=dict(color=NAVY, size=13), bordercolor=GARIS),
                      dragmode=False)
    return fig


def akurasi_html(tanggal_akhir, keandalan):
    batang = ''.join(
        f'<div class="batang"><span class="lb">{b["langkah"]} langkah (±{round(b["hari_rata"])} hari)</span>'
        f'<span class="bg"><span class="isi" style="width:{b["F1"] * 100:.0f}%"></span></span>'
        f'<span class="sk">{b["F1"]:.2f}</span></div>'
        for b in keandalan)
    return ('<div class="isi-lipat">'
            f'<p>Data satelit radar terakhir adalah {tanggal_akhir}. Semakin jauh tanggal yang dipilih, semakin banyak '
            'langkah tebakan berantai (sekitar 12 hari per langkah), dan semakin turun ketepatannya. Skor di bawah '
            'berasal dari uji pada data Maret–Juni 2026 (skor 1 = sempurna).</p>'
            f'{batang}'
            f'<p style="font-size:.9rem;color:#5f636c;margin-top:.6rem">Lebih dari {keandalan[-1]["langkah"]} langkah '
            f'(sekitar {round(keandalan[-1]["hari_rata"] / 30)} bulan setelah {tanggal_akhir}) belum pernah diuji.</p>'
            '</div>')


def tentang_html(n_scene, periode, langkah_hari, tahun_awal, tahun_akhir):
    return (
        '<div class="isi-lipat">'
        '<h4>Apa ini?</h4>'
        '<p>Aplikasi ini memperkirakan lokasi genangan air di Jabodetabek pada tanggal yang kamu pilih, lalu '
        'membandingkannya dengan kondisi yang biasa terjadi. Aplikasi ini dibuat sebagai bagian dari penelitian skripsi.</p>'
        '<h4>Dari mana datanya?</h4>'
        '<ul>'
        '<li><strong>Sentinel-1</strong>: satelit radar yang bisa melihat air di permukaan tanah, bahkan saat mendung. '
        'Dari sini kita tahu lokasi mana yang tergenang.</li>'
        '<li><strong>MODIS</strong>: satelit yang memotret kondisi tanaman dan kelembapan permukaan.</li>'
        '<li><strong>GPM</strong>: satelit pengukur curah hujan.</li>'
        '</ul>'
        f'<p>Data yang dipakai adalah {n_scene} pengamatan dari {periode}.</p>'
        '<h4>Bagaimana cara kerjanya?</h4>'
        '<ol>'
        '<li>Sebuah model kecerdasan buatan (ContextCNN) mempelajari bagaimana genangan berubah dari satu pengamatan '
        'satelit ke pengamatan berikutnya.</li>'
        f'<li>Untuk tanggal yang kamu pilih, model melihat tiga pengamatan satelit terakhir, lalu menebak kondisi '
        f'sekitar {langkah_hari} hari ke depan.</li>'
        f'<li>Kalau tanggalnya lebih jauh, tebakan itu dipakai lagi untuk menebak {langkah_hari} hari berikutnya, dan '
        'seterusnya. Karena itu, semakin jauh tanggalnya, semakin rendah keyakinannya.</li>'
        '<li>Curah hujan yang dipakai adalah rata-rata hujan yang biasa turun di bulan tersebut.</li>'
        '</ol>'
        '<h4>Cara membaca hasil</h4>'
        '<ul>'
        '<li><strong>Normal</strong>: luas genangan berada di kisaran yang biasa terjadi pada bulan yang sama di tahun '
        f'{tahun_awal}–{tahun_akhir}. <strong>Di atas normal</strong> berarti lebih luas dari biasanya, '
        '<strong>di bawah normal</strong> berarti lebih sempit.</li>'
        '<li><strong>Peta</strong>: warna biru tua menandai lokasi yang diperkirakan tergenang padahal biasanya kering. '
        'Biru muda adalah genangan yang memang biasa ada, misalnya tambak, waduk, dan sungai.</li>'
        '<li><strong>Seberapa yakin?</strong>: <em>Tinggi</em> untuk sekitar 2 minggu pertama setelah pengamatan '
        'terakhir, lalu turun menjadi <em>Sedang</em> dan <em>Rendah</em>. Lebih dari sekitar 3 bulan belum pernah diuji.</li>'
        '</ul>'
        '<h4>Apa yang perlu diingat?</h4>'
        '<ul>'
        '<li>Ini hasil penelitian, <strong>bukan peringatan banjir resmi</strong>. Untuk informasi resmi, rujuk BMKG dan '
        'BPBD setempat.</li>'
        '<li>Satu titik di peta mewakili area sekitar 50 × 50 meter, jadi genangan kecil di jalan atau gang tidak terlihat.</li>'
        '<li>Satelit radar sulit melihat genangan di bawah pepohonan atau di antara gedung tinggi.</li>'
        '<li>Model lebih banyak belajar dari lokasi yang biasa tergenang daripada dari curah hujan, sehingga hujan deras '
        'yang tiba-tiba belum tentu tercermin.</li>'
        '</ul>'
        '</div>'
    )