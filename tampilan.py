# topik1/tampilan.py
import io
import math

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
WARNA_STATUS = {'Aman': '#3f7d58', 'Waspada': '#c98a1b', 'Bahaya': '#b23b3b'}
SKALA_PROB = [[0.0, '#f3f2ee'], [0.45, '#9aa6bd'], [1.0, NAVY]]

IKON_STATUS = {
    'Aman': ('<path d="M12 1.8 3.8 5v6.2c0 5.1 3.5 9.6 8.2 11 4.7-1.4 8.2-5.9 8.2-11V5L12 1.8z" fill="#3f7d58"/>'
             '<path d="m8 12.2 2.8 2.8 5.2-6" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'),
    'Waspada': ('<path d="M10.3 3.4a2 2 0 0 1 3.4 0l8.6 15a2 2 0 0 1-1.7 3H3.4a2 2 0 0 1-1.7-3l8.6-15z" fill="#c98a1b"/>'
                '<rect x="11" y="8.2" width="2" height="7" rx="1" fill="#fff"/><circle cx="12" cy="18.1" r="1.2" fill="#fff"/>'),
    'Bahaya': ('<rect x="10.8" y="1.5" width="2.4" height="9.5" rx="1.2" fill="#b23b3b"/><circle cx="12" cy="13.9" r="1.4" fill="#b23b3b"/>'
               '<path d="M1.5 18.2c1.8 0 1.8-1.6 3.5-1.6s1.8 1.6 3.5 1.6 1.8-1.6 3.5-1.6 1.8 1.6 3.5 1.6 1.8-1.6 3.5-1.6 1.8 1.6 3.5 1.6" fill="none" stroke="#b23b3b" stroke-width="2" stroke-linecap="round"/>'
               '<path d="M1.5 22.2c1.8 0 1.8-1.6 3.5-1.6s1.8 1.6 3.5 1.6 1.8-1.6 3.5-1.6 1.8 1.6 3.5 1.6 1.8-1.6 3.5-1.6 1.8 1.6 3.5 1.6" fill="none" stroke="#b23b3b" stroke-width="2" stroke-linecap="round" opacity=".55"/>'),
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
    '.stat{padding:1.6rem 1rem;text-align:center;border-right:1px solid #e6e3dc}'
    '.stat:last-child{border-right:none}'
    '.stat-label{font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:#7a7f8a}'
    '.stat-nilai{font-family:"Source Serif 4",Georgia,serif;font-variant-numeric:oldstyle-nums;font-size:2.1rem;color:#1f2a44;margin-top:.45rem;line-height:1.15}'
    '.stat-ket{font-size:.78rem;color:#9a9ea8;margin-top:.25rem}'
    '.legenda{display:flex;justify-content:center;flex-wrap:wrap;gap:.3rem 1.1rem;font-size:.78rem;color:#7a7f8a;margin-top:.4rem}'
    '.kotak{display:inline-block;width:.7rem;height:.7rem;margin-right:.35rem;vertical-align:-0.05rem;border:1px solid #d7d3cb}'
    '.donat-wadah{max-width:240px;margin:0 auto;text-align:center}'
    '.donat-svg{width:100%;height:auto;display:block}'
    '.konteks{font-size:.88rem;color:#4b5160;line-height:1.65}'
    '.konteks b{color:#1f2a44}'
    '.catatan{background:#fbf6ea;border:1px solid #efe1bd;color:#7a5a12;border-radius:4px;padding:.55rem .75rem;font-size:.82rem;margin-top:.5rem;display:flex;gap:.45rem;align-items:flex-start}'
    '.kosong{border:1px dashed #d7d3cb;border-radius:6px;padding:2.6rem 1rem;text-align:center;color:#7a7f8a;background:#fbfaf7}'
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
        '<div class="sub-app">Simulasikan curah hujan atau cek tanggal tertentu, lalu lihat peta genangan yang '
        'diprediksi model untuk observasi Sentinel-1 berikutnya.</div>'
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


def donat_html(persen, status):
    p = min(max(float(persen), 0.0), 100.0)
    r, tebal = 92.0, 18.0
    keliling = 2 * math.pi * r
    busur = keliling * p / 100
    warna = WARNA_STATUS[status]
    svg = (
        f'<svg class="donat-svg" viewBox="0 0 240 240" role="img" aria-label="Tergenang {p:.2f} persen, status {status}">'
        f'<circle cx="120" cy="120" r="{r}" fill="none" stroke="{KERING}" stroke-width="{tebal}"/>'
        f'<circle cx="120" cy="120" r="{r}" fill="none" stroke="{AIR}" stroke-width="{tebal}" '
        f'stroke-dasharray="{busur:.3f} {keliling:.3f}" transform="rotate(-90 120 120)"/>'
        f'<g transform="translate(96 58) scale(2)">{IKON_STATUS[status]}</g>'
        f'<text x="120" y="140" text-anchor="middle" font-size="20" font-weight="600" fill="{warna}" '
        f'font-family="Source Serif 4, Georgia, serif">{status}</text>'
        f'<text x="120" y="162" text-anchor="middle" font-size="12.5" fill="#4b5160" font-family="Nunito Sans, sans-serif">'
        f'{p:.2f}% tergenang</text>'
        '</svg>'
    )
    legenda = legenda_html([(AIR, f'Tergenang {p:.2f}%'), (KERING, f'Tidak tergenang {100 - p:.2f}%')])
    return f'<div class="donat-wadah">{svg}{legenda}</div>'


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