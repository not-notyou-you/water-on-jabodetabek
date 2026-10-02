# tampilan.py
import io
import math
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

NAVY = '#1f2a44'
AIR = '#4a6fa5'
KERING = '#ece9e2'
LAUT = '#d5dce5'
GARIS = '#e6e3dc'
TEKS_REDUP = '#7a7f8a'
WARNA_BANJIR = [KERING, AIR, LAUT]
WARNA_PERUBAHAN = [KERING, NAVY, '#9db3d1', '#c98a1b', LAUT]
WARNA_ANOMALI = [KERING, '#9db3d1', NAVY, '#c98a1b', LAUT]
WARNA_KATEGORI = {'Di bawah normal': '#3f7d58', 'Normal': '#5b6b84', 'Di atas normal': '#b23b3b'}
WARNA_KEANDALAN = {3: '#3f7d58', 2: '#c98a1b', 1: '#b23b3b', 0: '#7a7f8a'}
SKALA_PROB = [[0.0, '#f3f2ee'], [0.45, '#9aa6bd'], [1.0, NAVY]]

IKON_KATEGORI = {
    'Di bawah normal': '<circle cx="12" cy="12" r="10" fill="{w}"/><path d="M12 6.5v10M7.5 12.5l4.5 4.5 4.5-4.5" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
    'Normal': '<circle cx="12" cy="12" r="10" fill="{w}"/><path d="M7.5 9.8h9M7.5 14.2h9" stroke="#fff" stroke-width="2.2" stroke-linecap="round"/>',
    'Di atas normal': '<circle cx="12" cy="12" r="10" fill="{w}"/><path d="M12 17.5v-10M7.5 11.5l4.5-4.5 4.5 4.5" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
}

IKON_UI = {
    'gelombang': ('<path d="M2 9c2.5 0 2.5-2 5-2s2.5 2 5 2 2.5-2 5-2 2.5 2 5 2M2 15c2.5 0 2.5-2 5-2s2.5 2 5 2 2.5-2 5-2 2.5 2 5 2" '
                  'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>'),
    'kalender': ('<rect x="3.5" y="5" width="17" height="15" rx="2" fill="none" stroke="currentColor" stroke-width="1.8"/>'
                 '<path d="M3.5 10h17M8 3v4M16 3v4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>'),
    'satelit': ('<path d="M13 7l4 4-6 6-4-4 6-6z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>'
                '<path d="M15 5l2-2 4 4-2 2M9 15l-2 2M5 19l2-2M4 13a7 7 0 0 0 7 7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>'),
    'info': ('<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/>'
             '<path d="M12 11v6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="12" cy="7.8" r="1.1" fill="currentColor"/>'),
    'kunci': ('<circle cx="8" cy="15" r="4" fill="none" stroke="currentColor" stroke-width="1.8"/>'
              '<path d="M11 12l9-9M17 6l3 3M15 8l2 2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>'),
    'peringatan': ('<path d="M10.3 3.4a2 2 0 0 1 3.4 0l8.6 15a2 2 0 0 1-1.7 3H3.4a2 2 0 0 1-1.7-3l8.6-15z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>'
                   '<path d="M12 9v5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="12" cy="17.3" r="1.1" fill="currentColor"/>'),
}

CSS = (
    '<style>'
    "@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');"
    'html,body,[class*="css"],.stMarkdown,.stTextInput,.stSlider,.stButton,.stTabs,.stDateInput{font-family:"Nunito Sans",system-ui,sans-serif}'
    '.block-container{max-width:1120px;padding-top:5rem;padding-bottom:4rem}'
    'h1,h2,h3{font-family:"Source Serif 4",Georgia,serif !important;color:#1f2a44 !important;font-weight:600 !important}'
    '[data-testid="stVerticalBlockBorderWrapper"]{background:#ffffff;border-color:#e6e3dc !important;border-radius:6px}'
    '.stButton>button,.stDownloadButton>button{border-radius:4px;letter-spacing:.03em;font-weight:600}'
    '.stButton>button[kind="primary"]{background:#1f2a44;border-color:#1f2a44;padding:.6rem 1.6rem}'
    '.stButton>button[kind="primary"]:hover{background:#2c3a5c;border-color:#2c3a5c}'
    '.stTabs [data-baseweb="tab-list"]{gap:1.5rem;border-bottom:1px solid #e6e3dc}'
    '.stTabs [data-baseweb="tab"]{font-size:.8rem;letter-spacing:.14em;text-transform:uppercase;color:#7a7f8a;padding:.6rem 0}'
    '.stTabs [aria-selected="true"]{color:#1f2a44}'
    '.alis{font-size:.72rem;letter-spacing:.22em;text-transform:uppercase;color:#7a7f8a;margin-bottom:.2rem}'
    '.judul-app{font-family:"Source Serif 4",Georgia,serif;font-size:2.5rem;line-height:1.1;color:#1f2a44;margin:0 0 .4rem 0;font-weight:600}'
    '.sub-app{color:#4b5160;font-size:1rem;max-width:720px;margin-bottom:.4rem}'
    '.riwayat{color:#7a7f8a;font-size:.85rem;display:flex;align-items:center;gap:.4rem;margin-bottom:1rem}'
    '.ikon-ui{display:inline-block;width:1.05em;height:1.05em;vertical-align:-0.17em;flex:none}'
    '.label-kecil{font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:#7a7f8a;text-align:center;margin:.2rem 0 .5rem 0}'
    '.rentang{font-size:.74rem;color:#9a9ea8;margin:-.9rem 0 .7rem 0}'
    '.info-tanggal{font-size:.88rem;color:#4b5160;display:flex;align-items:center;gap:.4rem;margin:.2rem 0 .6rem 0}'
    '.statistik{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));border-top:1px solid #e6e3dc;border-bottom:1px solid #e6e3dc;margin:1.2rem 0 1.4rem 0}'
    '.stat{padding:1.2rem 1rem;text-align:center;border-right:1px solid #e6e3dc}'
    '.stat:last-child{border-right:none}'
    '.stat-label{font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:#7a7f8a}'
    '.stat-nilai{font-family:"Source Serif 4",Georgia,serif;font-variant-numeric:oldstyle-nums;font-size:2.1rem;color:#1f2a44;margin-top:.45rem;line-height:1.15}'
    '.stat-ket{font-size:.78rem;color:#9a9ea8;margin-top:.25rem}'
    '.legenda{display:flex;justify-content:center;flex-wrap:wrap;gap:.3rem 1.1rem;font-size:.78rem;color:#7a7f8a;margin-top:.4rem}'
    '.kotak{display:inline-block;width:.7rem;height:.7rem;margin-right:.35rem;vertical-align:-0.05rem;border:1px solid #d7d3cb}'
    '.donat-wadah{max-width:240px;margin:0 auto;text-align:center}'
    '.donat-svg{width:100%;height:auto;display:block}'
    '.pil{display:inline-block;font-size:.72rem;padding:.12rem .5rem;border:1px solid;border-radius:999px;margin-right:.35rem;font-weight:600}'
    '.grid-tgl{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.8rem;margin:1rem 0 1.4rem 0}'
    '.kartu-tgl{background:#fff;border:1px solid #e6e3dc;border-radius:6px;padding:1rem 1.1rem}'
    '.kt-tgl{font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:#7a7f8a}'
    '.kt-luas{font-family:"Source Serif 4",Georgia,serif;font-variant-numeric:oldstyle-nums;font-size:2rem;color:#1f2a44;margin-top:.35rem;line-height:1.1}'
    '.kt-luas span{font-size:1rem;color:#4b5160}'
    '.kt-sub{font-size:.76rem;color:#9a9ea8;margin-top:.15rem}'
    '.kt-kat{display:flex;align-items:center;gap:.4rem;font-weight:700;margin-top:.7rem;font-size:.95rem}'
    '.kt-pil{margin-top:.65rem}'
    '.penjelasan{display:grid;gap:.45rem;font-size:.84rem;color:#4b5160}'
    '.penjelasan>div{display:flex;gap:.5rem;align-items:flex-start}'
    '.pj-cat{color:#7a7f8a;font-size:.8rem;margin-top:.2rem}'
    '.konteks{font-size:.88rem;color:#4b5160;line-height:1.65}'
    '.konteks b{color:#1f2a44}'
    '.catatan{background:#fbf6ea;border:1px solid #efe1bd;color:#7a5a12;border-radius:4px;padding:.55rem .75rem;font-size:.82rem;margin-top:.5rem;display:flex;gap:.45rem;align-items:flex-start}'
    '.kosong{border:1px dashed #d7d3cb;border-radius:6px;padding:2.6rem 1rem;text-align:center;color:#7a7f8a;background:#fbfaf7}'
    '.tabel-wadah{overflow-x:auto;border:1px solid #e6e3dc;border-radius:6px;background:#fff}'
    '.tabel-kanal{width:100%;border-collapse:collapse;font-size:.84rem;color:#4b5160}'
    '.tabel-kanal th{text-align:left;font-size:.66rem;letter-spacing:.14em;text-transform:uppercase;color:#7a7f8a;font-weight:600;padding:.6rem .8rem;border-bottom:1px solid #e6e3dc;background:#fbfaf7}'
    '.tabel-kanal td{padding:.55rem .8rem;border-bottom:1px solid #f0ede6;vertical-align:top}'
    '.tabel-kanal tr:last-child td{border-bottom:none}'
    '.tabel-kanal code{background:#f3f2ee;color:#1f2a44;padding:.05rem .3rem;border-radius:3px;font-size:.8rem;white-space:nowrap}'
    '@media (max-width:640px){.judul-app{font-size:1.9rem}.stat{border-right:none;border-bottom:1px solid #e6e3dc}.stat:last-child{border-bottom:none}.stat-nilai{font-size:1.8rem}}'
    '</style>'
)


def ikon_ui(nama, warna=TEKS_REDUP):
    return (f'<svg class="ikon-ui" viewBox="0 0 24 24" style="color:{warna}" aria-hidden="true">'
            f'{IKON_UI[nama]}</svg>')


def kepala_html(meta_teks_riwayat):
    return (
        f'<div class="alis">{ikon_ui("gelombang", "#7a7f8a")} Prediksi genangan multisensor</div>'
        '<div class="judul-app">Water on Jabodetabek</div>'
        '<div class="sub-app">Pilih satu sampai lima tanggal, tentukan sumber curah hujan, lalu lihat peta genangan '
        'yang diprediksi model beserta perbandingannya dengan kondisi normal bulan yang sama.</div>'
        f'<div class="riwayat">{ikon_ui("satelit")}<span>{meta_teks_riwayat}</span></div>'
    )


def label_kecil(teks):
    return f'<div class="label-kecil">{teks}</div>'


def statistik_html(daftar):
    isi = ''.join(
        f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-nilai">{nilai}</div>'
        f'<div class="stat-ket">{ket}</div></div>'
        for label, nilai, ket in daftar
    )
    return f'<div class="statistik">{isi}</div>'


def legenda_html(daftar):
    isi = ''.join(f'<span><span class="kotak" style="background:{w}"></span>{l}</span>' for w, l in daftar)
    return f'<div class="legenda">{isi}</div>'


def catatan_html(teks):
    return f'<div class="catatan">{ikon_ui("peringatan", "#a07614")}<span>{teks}</span></div>'


def ikon_kategori(nama, ukuran=22):
    w = WARNA_KATEGORI[nama]
    return (f'<svg width="{ukuran}" height="{ukuran}" viewBox="0 0 24 24" aria-hidden="true" style="flex:none">'
            f'{IKON_KATEGORI[nama].replace("{w}", w)}</svg>')


def pil(teks, warna):
    return (f'<span class="pil" style="color:{warna};border-color:{warna}33;background:{warna}12">{teks}</span>')


def kartu_tanggal_html(daftar):
    isi = []
    for h in daftar:
        k = h['kategori']
        kd = h['keandalan']
        perubahan = h['perubahan_km2']
        isi.append(
            '<div class="kartu-tgl">'
            f'<div class="kt-tgl">{h["judul"]}</div>'
            f'<div class="kt-luas">{h["luas_teks"]}<span> km²</span></div>'
            f'<div class="kt-sub">{"+" if perubahan >= 0 else "−"}{h["perubahan_teks"]} km² dari observasi terakhir</div>'
            f'<div class="kt-kat">{ikon_kategori(k["nama"])}<span style="color:{WARNA_KATEGORI[k["nama"]]}">{k["nama"]}</span></div>'
            f'<div class="kt-sub">normal bulan ini {h["normal_teks"]} km²</div>'
            f'<div class="kt-pil">{pil("Keandalan " + kd["label"].lower(), WARNA_KEANDALAN[kd["tingkat"]])}'
            f'{pil(str(h["langkah_ke"]) + " langkah", TEKS_REDUP)}</div>'
            '</div>'
        )
    return f'<div class="grid-tgl">{"".join(isi)}</div>'


def penjelasan_html():
    return (
        '<div class="penjelasan">'
        f'<div>{ikon_kategori("Di bawah normal", 18)}<span><b>Di bawah normal</b>: luas genangan lebih kecil dari '
        '25% kejadian historis pada bulan yang sama.</span></div>'
        f'<div>{ikon_kategori("Normal", 18)}<span><b>Normal</b>: berada di antara persentil 25 dan 75 historis bulan '
        'yang sama.</span></div>'
        f'<div>{ikon_kategori("Di atas normal", 18)}<span><b>Di atas normal</b>: lebih luas dari 75% kejadian '
        'historis bulan yang sama.</span></div>'
        '<div class="pj-cat">Tanggal jauh diprediksi berantai: hasil satu langkah (sekitar 12 hari) menjadi riwayat '
        'langkah berikutnya, sehingga keandalan menurun seiring jumlah langkah.</div>'
        '</div>'
    )


def konteks_html(baris):
    return '<div class="konteks">' + '<br>'.join(baris) + '</div>'


def _tata_letak(fig, rasio, latar='#ffffff'):
    fig.update_layout(
        height=430, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        plot_bgcolor=latar, paper_bgcolor='#ffffff',
        xaxis=dict(visible=False, constrain='domain'),
        yaxis=dict(visible=False, scaleanchor='x', scaleratio=rasio, constrain='domain'),
        hoverlabel=dict(bgcolor='#ffffff', font=dict(color=NAVY, size=12), bordercolor=GARIS),
    )
    return fig


def figur_kelas(kelas, lon, lat, rasio, warna, label):
    n = len(warna)
    skala = []
    for i, w in enumerate(warna):
        skala += [[i / n, w], [(i + 1) / n, w]]
    teks = np.asarray(label, dtype=object)[kelas]
    fig = go.Figure(go.Heatmap(
        z=kelas, x=lon, y=lat, colorscale=skala, zmin=-0.5, zmax=n - 0.5, showscale=False,
        customdata=teks, hovertemplate='%{customdata}<br>%{x:.3f}° BT, %{y:.3f}°<extra></extra>',
    ))
    return _tata_letak(fig, rasio)


def figur_probabilitas(prob, lon, lat, rasio):
    laut = np.isnan(prob)
    z = np.where(laut, -0.1, prob)
    batas = 0.1 / 1.1
    skala = [[0.0, LAUT], [batas, LAUT]] + [[batas + (1 - batas) * t, w] for t, w in SKALA_PROB]
    skala[2][0] = batas + 1e-6
    teks = np.where(laut, 'Laut', np.char.add('Probabilitas ', np.char.mod('%.3f', np.nan_to_num(prob))))
    fig = go.Figure(go.Heatmap(
        z=z, x=lon, y=lat, colorscale=skala, zmin=-0.1, zmax=1, showscale=False, customdata=teks,
        hovertemplate='%{customdata}<br>%{x:.3f}° BT, %{y:.3f}°<extra></extra>',
    ))
    return _tata_letak(fig, rasio)


def _hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_kelas(kelas, warna):
    palet = np.array([_hex_rgb(w) for w in warna], np.uint8)
    return palet[kelas]


def rgb_probabilitas(prob):
    titik = np.array([s[0] for s in SKALA_PROB])
    warna = np.array([_hex_rgb(s[1]) for s in SKALA_PROB], np.float32)
    p = np.nan_to_num(np.clip(prob, 0, 1), nan=0.0)
    rgb = np.stack([np.interp(p, titik, warna[:, i]) for i in range(3)], axis=-1).astype(np.uint8)
    rgb[np.isnan(prob)] = _hex_rgb(LAUT)
    return rgb


def _huruf(ukuran):
    try:
        return ImageFont.load_default(size=ukuran)
    except TypeError:
        return ImageFont.load_default()


def png_laporan(panel, judul, subjudul, skala=2):
    gambar_panel = [Image.fromarray(rgb).resize((rgb.shape[1] * skala, rgb.shape[0] * skala), Image.NEAREST)
                    for rgb, _, _ in panel]
    jarak = 40
    lebar = sum(g.width for g in gambar_panel) + jarak * (len(panel) + 1)
    tinggi_peta = max(g.height for g in gambar_panel)
    tinggi = 110 + tinggi_peta + 30 + 26 * max(len(leg) for _, _, leg in panel) + 30
    kanvas = Image.new('RGB', (lebar, tinggi), '#f6f5f1')
    gambar = ImageDraw.Draw(kanvas)
    gambar.text((jarak, 24), judul, fill=NAVY, font=_huruf(26))
    gambar.text((jarak, 62), subjudul, fill='#4b5160', font=_huruf(15))
    x = jarak
    for g, (_, label, legenda) in zip(gambar_panel, panel):
        gambar.text((x, 92), label.upper(), fill=TEKS_REDUP, font=_huruf(13))
        kanvas.paste(g, (x, 112))
        gambar.rectangle([x - 1, 111, x + g.width, 112 + g.height], outline=GARIS)
        y = 112 + g.height + 18
        for w, l in legenda:
            gambar.rectangle([x, y, x + 14, y + 14], fill=w, outline='#cfcac0')
            gambar.text((x + 22, y - 1), l, fill='#4b5160', font=_huruf(13))
            y += 26
        x += g.width + jarak
    buf = io.BytesIO()
    kanvas.save(buf, format='PNG')
    return buf.getvalue()


def figur_tren(daftar, observasi, normal_bulanan):
    fig = go.Figure()
    xs = [d for d, _, _ in normal_bulanan]
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[p75 for _, _, p75 in normal_bulanan] + [p25 for _, p25, _ in normal_bulanan][::-1],
                             fill='toself', fillcolor='rgba(91,107,132,0.13)', line=dict(width=0), hoverinfo='skip',
                             name='Rentang normal (P25–P75)'))
    fig.add_trace(go.Scatter(x=[observasi[0]], y=[observasi[1]], mode='markers', name='Observasi terakhir',
                             marker=dict(size=11, color='#ffffff', line=dict(color=NAVY, width=2)),
                             hovertemplate='Observasi %{x|%d %b %Y}<br>%{y:.1f} km²<extra></extra>'))
    fig.add_trace(go.Scatter(x=[observasi[0]] + [h['tanggal'] for h in daftar],
                             y=[observasi[1]] + [h['luas_km2'] for h in daftar], mode='lines',
                             line=dict(color=NAVY, width=1.5, dash='dot'), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=[h['tanggal'] for h in daftar], y=[h['luas_km2'] for h in daftar], mode='markers',
                             name='Prediksi',
                             marker=dict(size=13, color=[WARNA_KATEGORI[h['kategori']['nama']] for h in daftar],
                                         line=dict(color='#ffffff', width=1.5)),
                             customdata=[[h['kategori']['nama'], h['keandalan']['label']] for h in daftar],
                             hovertemplate='%{x|%d %b %Y}<br>%{y:.1f} km²<br>%{customdata[0]}<br>Keandalan %{customdata[1]}<extra></extra>'))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                      legend=dict(orientation='h', y=-0.18, x=0, font=dict(size=11)),
                      yaxis=dict(title='Luas genangan (km²)', gridcolor='#efede7', zeroline=False),
                      xaxis=dict(gridcolor='#efede7'), hoverlabel=dict(bgcolor='#ffffff', font=dict(color=NAVY)))
    return fig


def figur_keandalan(keandalan):
    fig = go.Figure()
    for mode, warna, nama in (('resmi', NAVY, 'Hujan aktual (GPM)'), ('klimatologi', '#9db3d1', 'Hujan rata-rata bulanan')):
        t = keandalan[mode]
        fig.add_trace(go.Scatter(x=[b['langkah'] for b in t], y=[b['F1'] for b in t], mode='lines+markers', name=nama,
                                 line=dict(color=warna, width=2), marker=dict(size=8),
                                 error_y=dict(type='data', array=[b['F1_std'] for b in t], color=warna, thickness=1),
                                 customdata=[b['hari_rata'] for b in t],
                                 hovertemplate='Langkah %{x} (±%{customdata:.0f} hari)<br>F1 %{y:.3f}<extra></extra>'))
    t = keandalan['resmi']
    fig.add_trace(go.Scatter(x=[b['langkah'] for b in t], y=[b['F1_persistence'] for b in t], mode='lines+markers',
                             name='Persistence', line=dict(color='#b9b4aa', width=1.5, dash='dash'), marker=dict(size=6),
                             hovertemplate='Langkah %{x}<br>F1 %{y:.3f}<extra></extra>'))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                      legend=dict(orientation='h', y=-0.22, x=0, font=dict(size=11)),
                      yaxis=dict(title='F1', range=[0, 1], gridcolor='#efede7'),
                      xaxis=dict(title='Jumlah langkah prediksi berantai', dtick=1, gridcolor='#efede7'))
    return fig


def figur_akurasi_luas(scene):
    x = [datetime.strptime(s['tanggal'], '%Y%m%d') for s in scene]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[s['luas_aktual_km2'] for s in scene], mode='lines+markers', name='Aktual (SAR)',
                             line=dict(color='#7a7f8a', width=2)))
    fig.add_trace(go.Scatter(x=x, y=[s['luas_pred_km2'] for s in scene], mode='lines+markers', name='Prediksi',
                             line=dict(color=NAVY, width=2)))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                      legend=dict(orientation='h', y=-0.2, x=0), yaxis=dict(title='Luas genangan (km²)', gridcolor='#efede7'),
                      xaxis=dict(gridcolor='#efede7'))
    return fig