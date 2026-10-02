# topik1/tampilan.py
import io
import math

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

WARNA_BANJIR = ['#dcfce7', '#2563eb', '#e5e7eb']
WARNA_PERUBAHAN = ['#dcfce7', '#1d4ed8', '#93c5fd', '#f59e0b', '#e5e7eb']
WARNA_STATUS = {'Aman': '#16a34a', 'Waspada': '#f59e0b', 'Bahaya': '#dc2626'}
SKALA_PROB = [[0.0, '#f0f9ff'], [0.5, '#60a5fa'], [1.0, '#1e3a8a']]

IKON_STATUS = {
    'Aman': ('<path d="M12 1.8 3.8 5v6.2c0 5.1 3.5 9.6 8.2 11 4.7-1.4 8.2-5.9 8.2-11V5L12 1.8z" fill="#16a34a"/>'
             '<path d="m8 12.2 2.8 2.8 5.2-6" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'),
    'Waspada': ('<path d="M10.3 3.4a2 2 0 0 1 3.4 0l8.6 15a2 2 0 0 1-1.7 3H3.4a2 2 0 0 1-1.7-3l8.6-15z" fill="#f59e0b"/>'
                '<rect x="11" y="8.2" width="2" height="7" rx="1" fill="#fff"/><circle cx="12" cy="18.1" r="1.2" fill="#fff"/>'),
    'Bahaya': ('<rect x="10.8" y="1.5" width="2.4" height="9.5" rx="1.2" fill="#dc2626"/><circle cx="12" cy="13.9" r="1.4" fill="#dc2626"/>'
               '<path d="M1.5 18.2c1.8 0 1.8-1.6 3.5-1.6s1.8 1.6 3.5 1.6 1.8-1.6 3.5-1.6 1.8 1.6 3.5 1.6 1.8-1.6 3.5-1.6 1.8 1.6 3.5 1.6" fill="none" stroke="#dc2626" stroke-width="2" stroke-linecap="round"/>'
               '<path d="M1.5 22.2c1.8 0 1.8-1.6 3.5-1.6s1.8 1.6 3.5 1.6 1.8-1.6 3.5-1.6 1.8 1.6 3.5 1.6 1.8-1.6 3.5-1.6 1.8 1.6 3.5 1.6" fill="none" stroke="#dc2626" stroke-width="2" stroke-linecap="round" opacity=".55"/>'),
}

IKON_UI = {
    'luas': ('<path d="M3 7l6-3 6 3 6-3v13l-6 3-6-3-6 3V7z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>'
             '<path d="M9 4v13M15 7v13" stroke="currentColor" stroke-width="1.8"/>'),
    'perubahan': ('<path d="M4 17l5-5 4 4 7-8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
                  '<path d="M15 8h5v5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'),
    'persen': ('<path d="M12 3c3.5 4.4 6 7.8 6 11a6 6 0 0 1-12 0c0-3.2 2.5-6.6 6-11z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>'),
    'ambang': ('<path d="M4 20h16M6 20V10M12 20V4M18 20v-7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>'),
    'kalender': ('<rect x="3.5" y="5" width="17" height="15" rx="2" fill="none" stroke="currentColor" stroke-width="1.8"/>'
                 '<path d="M3.5 10h17M8 3v4M16 3v4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>'),
    'satelit': ('<path d="M13 7l4 4-6 6-4-4 6-6z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>'
                '<path d="M15 5l2-2 4 4-2 2M9 15l-2 2M5 19l2-2M4 13a7 7 0 0 0 7 7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>'),
    'info': ('<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/>'
             '<path d="M12 11v6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="12" cy="7.8" r="1.1" fill="currentColor"/>'),
}

CSS = (
    '<style>'
    '.blok-container{padding-top:1.6rem}'
    '.ikon-ui{display:inline-block;width:1.05em;height:1.05em;vertical-align:-0.17em;margin-right:.35rem}'
    '.baris-info{color:#374151;font-size:.95rem;margin:.1rem 0 .6rem 0}'
    '.donat-wadah{max-width:300px;margin:0 auto;text-align:center}'
    '.donat-svg{width:100%;height:auto;display:block}'
    '.donat-legenda{display:flex;flex-wrap:wrap;justify-content:center;gap:.4rem 1rem;font-size:.88rem;color:#374151;margin-top:.3rem}'
    '.kotak-warna{display:inline-block;width:.8rem;height:.8rem;border-radius:2px;margin-right:.35rem;vertical-align:-0.08rem;border:1px solid #d1d5db}'
    '.kartu-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.7rem}'
    '.kartu{border:1px solid #e5e7eb;border-radius:10px;padding:.75rem .9rem;background:#fff}'
    '.kartu-judul{font-size:.82rem;color:#6b7280;display:flex;align-items:center}'
    '.kartu-nilai{font-size:1.35rem;font-weight:700;color:#111827;margin-top:.15rem;line-height:1.25}'
    '.kartu-ket{font-size:.8rem;color:#6b7280;margin-top:.1rem}'
    '.kosong{border:1px dashed #cbd5e1;border-radius:12px;padding:2.5rem 1rem;text-align:center;color:#6b7280}'
    '.rentang{font-size:.78rem;color:#6b7280;margin:-.6rem 0 .5rem 0}'
    '</style>'
)


def ikon_ui(nama, warna='#2563eb'):
    return (f'<svg class="ikon-ui" viewBox="0 0 24 24" style="color:{warna}" aria-hidden="true">'
            f'{IKON_UI[nama]}</svg>')


def donat_html(persen, status):
    p = min(max(float(persen), 0.0), 100.0)
    r, tebal = 92.0, 22.0
    keliling = 2 * math.pi * r
    busur = keliling * p / 100
    warna = WARNA_STATUS[status]
    svg = (
        f'<svg class="donat-svg" viewBox="0 0 240 240" role="img" aria-label="Tergenang {p:.2f} persen, status {status}">'
        f'<circle cx="120" cy="120" r="{r}" fill="none" stroke="{WARNA_BANJIR[0]}" stroke-width="{tebal}"/>'
        f'<circle cx="120" cy="120" r="{r}" fill="none" stroke="{WARNA_BANJIR[1]}" stroke-width="{tebal}" '
        f'stroke-dasharray="{busur:.3f} {keliling:.3f}" transform="rotate(-90 120 120)"/>'
        f'<g transform="translate(96 60) scale(2)">{IKON_STATUS[status]}</g>'
        f'<text x="120" y="141" text-anchor="middle" font-size="19" font-weight="700" fill="{warna}" '
        f'font-family="sans-serif">{status}</text>'
        f'<text x="120" y="163" text-anchor="middle" font-size="13" fill="#374151" font-family="sans-serif">'
        f'{p:.2f}% tergenang</text>'
        '</svg>'
    )
    legenda = (
        '<div class="donat-legenda">'
        f'<span><span class="kotak-warna" style="background:{WARNA_BANJIR[1]}"></span>Tergenang <b>{p:.2f}%</b></span>'
        f'<span><span class="kotak-warna" style="background:{WARNA_BANJIR[0]}"></span>Tidak tergenang <b>{100 - p:.2f}%</b></span>'
        '</div>'
    )
    return f'<div class="donat-wadah">{svg}{legenda}</div>'


def kartu_html(daftar):
    isi = ''.join(
        f'<div class="kartu"><div class="kartu-judul">{ikon_ui(ikon)}{judul}</div>'
        f'<div class="kartu-nilai">{nilai}</div><div class="kartu-ket">{ket}</div></div>'
        for ikon, judul, nilai, ket in daftar
    )
    return f'<div class="kartu-grid">{isi}</div>'


def _tata_letak(fig, rasio, judul):
    fig.update_layout(
        title=dict(text=judul, x=0, font=dict(size=15)),
        height=540, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='#e5e7eb', paper_bgcolor='#ffffff',
        legend=dict(orientation='h', yanchor='top', y=-0.08, x=0, font=dict(size=12)),
        xaxis=dict(title='Bujur', constrain='domain', showgrid=False, ticksuffix='°'),
        yaxis=dict(title='Lintang', scaleanchor='x', scaleratio=rasio, constrain='domain', showgrid=False, ticksuffix='°'),
    )
    return fig


def figur_kelas(kelas, lon, lat, rasio, warna, label, judul):
    n = len(warna)
    skala = []
    for i, w in enumerate(warna):
        skala += [[i / n, w], [(i + 1) / n, w]]
    teks = np.asarray(label, dtype=object)[kelas]
    fig = go.Figure(go.Heatmap(
        z=kelas, x=lon, y=lat, colorscale=skala, zmin=-0.5, zmax=n - 0.5, showscale=False,
        customdata=teks, hovertemplate='Bujur %{x:.3f}°<br>Lintang %{y:.3f}°<br>%{customdata}<extra></extra>',
    ))
    for w, l in zip(warna, label):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=l,
                                 marker=dict(size=13, color=w, symbol='square', line=dict(color='#9ca3af', width=1))))
    return _tata_letak(fig, rasio, judul)


def figur_probabilitas(prob, lon, lat, rasio, threshold, judul):
    fig = go.Figure(go.Heatmap(
        z=prob, x=lon, y=lat, colorscale=SKALA_PROB, zmin=0, zmax=1,
        colorbar=dict(title='Prob.', thickness=12, len=0.75),
        hovertemplate='Bujur %{x:.3f}°<br>Lintang %{y:.3f}°<br>Probabilitas %{z:.3f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=f'Laut (abu). Threshold tergenang {threshold:.3f}',
                             marker=dict(size=13, color='#e5e7eb', symbol='square', line=dict(color='#9ca3af', width=1))))
    return _tata_letak(fig, rasio, judul)


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
    rgb[np.isnan(prob)] = _hex_rgb('#e5e7eb')
    return rgb


def _huruf(ukuran):
    try:
        return ImageFont.load_default(size=ukuran)
    except TypeError:
        return ImageFont.load_default()


def png_peta(rgb, judul, subjudul, legenda, skala=2):
    peta = Image.fromarray(rgb).resize((rgb.shape[1] * skala, rgb.shape[0] * skala), Image.NEAREST)
    lebar = max(peta.width + 40, 640)
    tinggi_legenda = 30 * math.ceil(len(legenda) / 3) + 20
    kanvas = Image.new('RGB', (lebar, peta.height + 90 + tinggi_legenda), 'white')
    gambar = ImageDraw.Draw(kanvas)
    gambar.text((20, 14), judul, fill='#111827', font=_huruf(20))
    gambar.text((20, 44), subjudul, fill='#4b5563', font=_huruf(14))
    x0 = (lebar - peta.width) // 2
    kanvas.paste(peta, (x0, 75))
    gambar.rectangle([x0 - 1, 74, x0 + peta.width, 75 + peta.height], outline='#9ca3af')
    huruf = _huruf(14)
    y = 75 + peta.height + 18
    kolom = (lebar - 40) // 3
    for i, (w, l) in enumerate(legenda):
        x = 20 + (i % 3) * kolom
        yy = y + (i // 3) * 30
        gambar.rectangle([x, yy, x + 16, yy + 16], fill=w, outline='#9ca3af')
        gambar.text((x + 24, yy), l, fill='#111827', font=huruf)
    buf = io.BytesIO()
    kanvas.save(buf, format='PNG')
    return buf.getvalue()