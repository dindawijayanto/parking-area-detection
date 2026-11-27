import os
import sys
from pathlib import Path

import cv2
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import win32com.client
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(r"c:\Users\kuahs\Documents\projek pcd")
LOGO_UB = Path(r"c:\Users\kuahs\Documents\pcd\dataset_tomat\logo_ub.png")

ASSETS_DIR = ROOT / "results" / "laporan_assets"
PROOFS_DIR = ROOT / "results" / "laporan_proofs"

OUT_DOCX = ROOT / "Laporan_Proyek_PCD_Deteksi_Area_Parkir_TOBIAS_245150200111076.docx"
OUT_PDF = ROOT / "Laporan_Proyek_PCD_Deteksi_Area_Parkir_TOBIAS_245150200111076.pdf"

FIG1_PATH = ASSETS_DIR / "fig1_dataset_samples.png"
FIG2_PATH = ROOT / "results" / "rf_vs_xgb_v2" / "v2_feature_importance.png"
FIG3_PATH = ROOT / "results" / "rf_vs_xgb_v2" / "v2_test_comparison_grid.png"
FIG4_PATH = ROOT / "results" / "optuna_rf_xgb" / "optuna_history.png"
FIG5_PATH = ROOT / "results" / "optuna_rf_xgb" / "optuna_test_comparison_grid.png"
FIG6_PATH = ROOT / "results" / "optuna_rf_xgb" / "optuna_bev_free_spaces_grid.png"


def generate_fig1_dataset_samples():
    """Buat kolase 2x2 dari 4 frame overlay anotasi asli mewakili 4 kondisi pencahayaan."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    samples = [
        (
            ROOT / "raw + dl" / "09-12-11-2026-morning-top-keiza" / "09-12-11-2026-morning-top-keiza_00-00_overlay.jpg",
            "(a) Pagi Cerah (09-12-11, Test)",
        ),
        (
            ROOT / "raw + dl" / "03-04-11-2026-afternoon-top-dinda" / "03-04-11-2026-afternoon-top-dinda_00-30_overlay.jpg",
            "(b) Siang Cerah (03-04-11, Test)",
        ),
        (
            ROOT / "raw + dl" / "08-10-11-2026-afternoon-rainy-top-tobi" / "08-10-11-2026-afternoon-rainy-top-tobi_00-30_overlay.jpg",
            "(c) Sore Hujan (08-10-11, Validation)",
        ),
        (
            ROOT / "raw + dl" / "04-04-11-2026-evening-top-keiza" / "04-04-11-2026-evening-top-keiza_00-30_overlay.jpg",
            "(d) Senja Gelap (04-04-11, Test)",
        ),
    ]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04, wspace=0.06, hspace=0.18)

    for ax, (img_path, title) in zip(axes.flat, samples):
        if not img_path.exists():
            raise FileNotFoundError(f"Overlay dataset tidak ditemukan: {img_path}")
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        target_ratio = 16.0 / 9.0
        cur_ratio = w / float(h)
        if cur_ratio < target_ratio - 0.05:
            # Frame potret/kotak (mis. 08-10-11): crop rapi ke rasio 16:9 berpusat di area parkir utama
            new_h = int(round(w / target_ratio))
            y0 = min(max(int(round(h * 0.28)), 0), h - new_h)
            rgb = rgb[y0 : y0 + new_h, :]
        elif cur_ratio > target_ratio + 0.05:
            new_w = int(round(h * target_ratio))
            x0 = (w - new_w) // 2
            rgb = rgb[:, x0 : x0 + new_w]
        rgb = cv2.resize(rgb, (960, 540), interpolation=cv2.INTER_AREA)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=11, color="black", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)

    fig.savefig(str(FIG1_PATH), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[+] Kolase dataset ditulis: {FIG1_PATH}")


def set_run_font(run, name="Times New Roman", size=12, bold=False, italic=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:cs"), name)
    rFonts.set(qn("w:eastAsia"), name)
    return run


def format_paragraph(
    p,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    space_before=0,
    space_after=4,
    line_spacing=1.15,
    keep_with_next=False,
):
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line_spacing
    pf.keep_with_next = keep_with_next
    return p


def set_cell_white(cell):
    tcPr = cell._element.get_or_add_tcPr()
    for child in list(tcPr):
        if child.tag == qn("w:shd"):
            tcPr.remove(child)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="FFFFFF"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=55, bottom=55, left=80, right=80):
    tcPr = cell._element.get_or_add_tcPr()
    for child in list(tcPr):
        if child.tag == qn("w:tcMar"):
            tcPr.remove(child)
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f"</w:tcMar>"
    )
    tcPr.append(tcMar)


def set_cell_width(cell, width_inches):
    cell.width = Inches(width_inches)
    tcPr = cell._element.get_or_add_tcPr()
    for child in list(tcPr):
        if child.tag == qn("w:tcW"):
            tcPr.remove(child)
    tcW = OxmlElement("w:tcW")
    tcW.set(qn("w:w"), str(int(round(width_inches * 1440))))
    tcW.set(qn("w:type"), "dxa")
    tcPr.append(tcW)


def set_table_borders_black(table, sz="4"):
    tblPr = table._element.tblPr
    for child in list(tblPr):
        if child.tag in (qn("w:tblBorders"), qn("w:tblLayout")):
            tblPr.remove(child)
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'
        f'<w:left w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'
        f'<w:right w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'
        f'<w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'
        f'<w:insideV w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'
        f"</w:tblBorders>"
    )
    tblPr.append(borders)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)


def set_row_cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))


def set_row_header(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:tblHeader"))


def add_page_number_footer(section):
    section.different_first_page_header_footer = True
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)

    r_begin = p.add_run()
    set_run_font(r_begin, size=10)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    r_begin._r.append(fld_begin)

    r_instr = p.add_run()
    set_run_font(r_instr, size=10)
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    r_instr._r.append(instr)

    r_sep = p.add_run()
    set_run_font(r_sep, size=10)
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    r_sep._r.append(fld_sep)

    r_end = p.add_run()
    set_run_font(r_end, size=10)
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    r_end._r.append(fld_end)


def build_report():
    for required in [LOGO_UB, FIG2_PATH, FIG3_PATH, FIG4_PATH, FIG5_PATH, FIG6_PATH]:
        if not required.exists():
            raise FileNotFoundError(f"Aset wajib tidak ditemukan: {required}")

    generate_fig1_dataset_samples()

    doc = Document()

    # Konfigurasi gaya Normal global ke Times New Roman 12 pt hitam
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    rPr_norm = normal_style.element.get_or_add_rPr()
    rFonts_norm = rPr_norm.find(qn("w:rFonts"))
    if rFonts_norm is None:
        rFonts_norm = OxmlElement("w:rFonts")
        rPr_norm.insert(0, rFonts_norm)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts_norm.set(qn(attr), "Times New Roman")

    # Ukuran kertas A4 dan margin 1 inci di keempat sisi
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.footer_distance = Inches(0.55)
        add_page_number_footer(section)

    def add_para(
        text="",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        bold=False,
        italic=False,
        size=12,
        space_before=0,
        space_after=4,
        line_spacing=1.15,
        keep_with_next=False,
    ):
        p = doc.add_paragraph()
        format_paragraph(
            p,
            align=align,
            space_before=space_before,
            space_after=space_after,
            line_spacing=line_spacing,
            keep_with_next=keep_with_next,
        )
        if text:
            r = p.add_run(text)
            set_run_font(r, size=size, bold=bold, italic=italic)
        return p

    def add_h1(text):
        return add_para(
            text,
            align=WD_ALIGN_PARAGRAPH.LEFT,
            bold=True,
            size=13,
            space_before=12,
            space_after=4,
            line_spacing=1.15,
            keep_with_next=True,
        )

    def add_h2(text):
        return add_para(
            text,
            align=WD_ALIGN_PARAGRAPH.LEFT,
            bold=True,
            size=12,
            space_before=8,
            space_after=3,
            line_spacing=1.15,
            keep_with_next=True,
        )

    def add_image_with_caption(img_path, caption, width_in=6.0, space_after=8):
        p_img = doc.add_paragraph()
        format_paragraph(
            p_img,
            align=WD_ALIGN_PARAGRAPH.CENTER,
            space_before=6,
            space_after=2,
            line_spacing=1.0,
            keep_with_next=True,
        )
        run_img = p_img.add_run()
        run_img.add_picture(str(img_path), width=Inches(width_in))

        add_para(
            caption,
            align=WD_ALIGN_PARAGRAPH.CENTER,
            bold=False,
            italic=False,
            size=10,
            space_before=2,
            space_after=space_after,
            line_spacing=1.1,
        )

    def add_bw_table(
        caption,
        headers,
        rows,
        col_widths,
        alignments=None,
        font_size=9.5,
        has_header=True,
        keep_together=True,
    ):
        if caption:
            add_para(
                caption,
                align=WD_ALIGN_PARAGRAPH.CENTER,
                bold=False,
                italic=False,
                size=10,
                space_before=6,
                space_after=3,
                line_spacing=1.1,
                keep_with_next=True,
            )

        total_rows = len(rows) + (1 if has_header else 0)
        tbl = doc.add_table(rows=total_rows, cols=len(col_widths))
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        set_table_borders_black(tbl, sz="4")

        row_offset = 0
        if has_header:
            hdr_row = tbl.rows[0]
            set_row_cant_split(hdr_row)
            set_row_header(hdr_row)
            for c_idx, text in enumerate(headers):
                cell = hdr_row.cells[c_idx]
                set_cell_width(cell, col_widths[c_idx])
                set_cell_white(cell)
                set_cell_margins(cell, top=55, bottom=55, left=60, right=60)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = cell.paragraphs[0]
                format_paragraph(
                    p,
                    align=WD_ALIGN_PARAGRAPH.CENTER,
                    space_before=0,
                    space_after=0,
                    line_spacing=1.05,
                    keep_with_next=keep_together,
                )
                r = p.add_run(str(text))
                set_run_font(r, size=font_size, bold=True)
            row_offset = 1

        for r_idx, row_vals in enumerate(rows):
            row = tbl.rows[r_idx + row_offset]
            set_row_cant_split(row)
            is_last_row = r_idx == (len(rows) - 1)
            for c_idx, val in enumerate(row_vals):
                cell = row.cells[c_idx]
                set_cell_width(cell, col_widths[c_idx])
                set_cell_white(cell)
                set_cell_margins(cell, top=48, bottom=48, left=60, right=60)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = cell.paragraphs[0]
                col_align = WD_ALIGN_PARAGRAPH.LEFT
                if alignments and c_idx < len(alignments):
                    col_align = alignments[c_idx]
                format_paragraph(
                    p,
                    align=col_align,
                    space_before=0,
                    space_after=0,
                    line_spacing=1.05,
                    keep_with_next=(keep_together and not is_last_row),
                )
                r = p.add_run(str(val))
                set_run_font(r, size=font_size, bold=False)

        # Jarak bawah tabel yang rapi
        add_para("", size=4, space_before=0, space_after=4, line_spacing=1.0)
        return tbl

    # =========================================================================
    # HALAMAN SAMPUL (COVER PAGE — GAYA FILKOM UB)
    # =========================================================================
    add_para(
        "LAPORAN PROYEK PENGOLAHAN CITRA DIGITAL",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        bold=True,
        size=14,
        space_before=18,
        space_after=8,
        line_spacing=1.15,
    )
    add_para(
        "DETEKSI AREA PARKIR KOSONG SEPEDA MOTOR BERBASIS SEGMENTASI PIKSEL FITUR CITRA DIGITAL MENGGUNAKAN RANDOM FOREST, XGBOOST, TUNING OPTUNA, DAN PROYEKSI BIRD'S-EYE VIEW",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        bold=True,
        size=12,
        space_before=4,
        space_after=32,
        line_spacing=1.25,
    )

    p_logo = doc.add_paragraph()
    format_paragraph(p_logo, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=36, line_spacing=1.0)
    p_logo.add_run().add_picture(str(LOGO_UB), width=Inches(1.95))

    add_para("Disusun Oleh:", align=WD_ALIGN_PARAGRAPH.CENTER, bold=False, size=11, space_after=4)
    add_para(
        "TOBIAS ANDRA VALENTINO\nNIM 245150200111076",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        bold=True,
        size=12,
        space_after=36,
        line_spacing=1.25,
    )

    add_para("Dosen Pengampu:", align=WD_ALIGN_PARAGRAPH.CENTER, bold=False, size=11, space_after=4)
    add_para(
        "Dosen Pengampu Mata Kuliah Pengolahan Citra Digital",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        bold=False,
        size=11,
        space_after=64,
    )

    add_para(
        "Program Studi Teknik Informatika\nDepartemen Teknik Informatika\nFakultas Ilmu Komputer\nUniversitas Brawijaya\n2026",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        bold=True,
        size=12,
        space_before=12,
        space_after=0,
        line_spacing=1.25,
    )

    doc.add_page_break()

    # =========================================================================
    # BAB 1: PENDAHULUAN
    # =========================================================================
    add_h1("1. Pendahuluan")

    add_h2("1.1 Latar Belakang")
    add_para(
        "Pengelolaan kapasitas parkir sepeda motor di lingkungan kampus membutuhkan pemantauan ruang kosong yang akurat meskipun penataan kendaraan di lapangan tidak memiliki garis slot permanen. Kamera pengawas yang dipasang di posisi elevasi tinggi menghasilkan citra miring dengan distorsi perspektif yang kuat, variasi pencahayaan ekstrem mulai dari pagi cerah, siang terik, hujan sore, hingga senja redup, serta bayangan tajam dari bangunan sekitar. Kondisi tersebut membuat pendekatan segmentasi sederhana berbasis ambang batas warna tunggal mudah gagal membedakan lantai parkir yang ternaungi bayangan dari bodi sepeda motor yang berwarna gelap."
    )
    add_para(
        "Pada proyek kelompok Pengolahan Citra Digital pada repositori parking-area-detection bersama anggota tim Dinda, Keiza, dan Micheline, pendekatan baseline Naive Bayes yang sebelumnya dikembangkan hanya menghasilkan mIoU sebesar 0.6470 dengan kesalahan estimasi kapasitas rata-rata 12.81 ruang kosong per frame. Rendahnya performa tersebut disebabkan oleh asumsi independensi fitur pada Naive Bayes yang tidak mampu menangkap interaksi non-linear antara warna lokal, kontras bayangan, dan tekstur permukaan kendaraan. Oleh karena itu, bagian pekerjaan yang dikerjakan pada laporan ini berfokus pada kurasi dataset anotasi terstandar, rekayasa fitur pengolahan citra digital yang tahan terhadap pergeseran pencahayaan senja, pembandingan sistematis model Random Forest dan XGBoost beserta optimasi hiperparameter menggunakan Optuna, serta integrasi proyeksi Bird's-Eye View (BEV) untuk menghitung jumlah ruang parkir kosong secara terukur."
    )

    add_h2("1.2 Tujuan Proyek")
    add_para(
        "Proyek ini bertujuan membangun pipeline segmentasi area lantai parkir dan sepeda motor secara lokal menggunakan ekstraksi fitur pengolahan citra digital murni tanpa ketergantungan pada jaringan deep learning berat. Pelaksanaan proyek mencakup penyusunan dataset 111 frame tajam dari 13 video lintas 8 tanggal dengan skema pemisahan hari yang bebas kebocoran data, perancangan 26 fitur lokal yang tahan terhadap perubahan tingkat kepadatan parkir serta pencahayaan senja, komparasi terukur antara Random Forest dan XGBoost melalui tiga tahap eksperimen termasuk tuning Optuna, serta evaluasi langsung terhadap estimasi jumlah ruang parkir kosong pada bidang Bird's-Eye View (BEV)."
    )

    add_h2("1.3 Lingkungan Pengujian")
    add_para(
        "Seluruh proses ekstraksi fitur, pelatihan model, pencarian hiperparameter Optuna, pascapemrosesan morfologi, dan evaluasi proyeksi Bird's-Eye View dijalankan sepenuhnya secara lokal pada satu mesin kerja agar waktu latih, latensi inferensi, dan ukuran model dapat dibandingkan secara adil. Ringkasan perangkat keras dan perangkat lunak yang digunakan tercantum pada Tabel 1.1."
    )

    add_bw_table(
        caption="Tabel 1.1 Spesifikasi Lingkungan Perangkat Keras dan Perangkat Lunak",
        headers=["Komponen", "Spesifikasi yang Digunakan"],
        rows=[
            ("Sistem Operasi", "Windows 11 x64 (Build 26300)"),
            ("Prosesor", "16 Logical CPU Cores (Eksekusi Paralel Lokal)"),
            ("Bahasa Pemrograman", "Python 3.13.14"),
            ("Pustaka Pengolahan Citra", "OpenCV 5.0.0 dan NumPy 2.1.3"),
            ("Pustaka Pemodelan", "scikit-learn 1.5.2 (Random Forest) dan XGBoost 3.2.0"),
            ("Pustaka Optimasi", "Optuna 4.9.0 (Tree-structured Parzen Estimator)"),
            ("Perangkat Anotasi", "Aplikasi Web Masking Parkir dengan Penyimpanan IndexedDB"),
        ],
        col_widths=[2.15, 4.10],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
        font_size=9.5,
    )

    # =========================================================================
    # BAB 2: METODOLOGI PENGERJAAN
    # =========================================================================
    add_h1("2. Metodologi Pengerjaan")

    add_h2("2.1 Kurasi Dataset dan Skema Pembagian Anti-Leakage")
    add_para(
        "Sumber data berasal dari 13 rekaman video pengawasan area parkir sepeda motor pada 8 tanggal berbeda dengan resolusi seragam 960 × 540 piksel yang disampel setiap interval 15 detik. Dari seluruh kandidat frame, satu frame buram pada rekaman sore tanggal 17 November dikeluarkan sehingga tersisa tepat 111 frame tajam. Anotasi ground truth dilakukan menggunakan alat web Masking Parkir yang telah diperbaiki sinkronisasi dimensi mask pada IndexedDB agar tidak terjadi kekosongan label saat pemuatan ulang. Setiap piksel diklasifikasikan ke dalam tiga kelas yaitu nilai 0 untuk lantai parkir kosong, nilai 1 untuk sepeda motor pada dua baris depan yang menjadi wilayah pengamatan utama, dan nilai 255 untuk area luar pengamatan yang diabaikan selama pelatihan maupun evaluasi."
    )
    add_para(
        "Pembagian data dilakukan berdasarkan kelompok tanggal perekaman, bukan acak per frame, untuk mencegah kebocoran data akibat kemiripan latar dan posisi kendaraan antarframe dalam satu hari yang sama. Sebanyak 67 frame dari 4 tanggal digunakan sebagai data latih, 18 frame dari 2 tanggal yang mencakup kondisi hujan sore dan pagi cerah digunakan sebagai data validasi untuk pemilihan model serta penyetelan ambang pascapemrosesan, dan 26 frame dari 2 tanggal terpisah digunakan sebagai data uji murni. Pemilihan hari uji pada tanggal 4 November dan 12 November disamakan persis dengan konfigurasi baseline Naive Bayes repositori tim agar perbandingan metrik bersifat apple-to-apple. Komposisi rinci dataset disajikan pada Tabel 2.1, sementara contoh tampilan overlay anotasi dari empat kondisi pencahayaan ditunjukkan pada Gambar 1."
    )

    add_bw_table(
        caption="Tabel 2.1 Distribusi Dataset 111 Frame Berdasarkan Skema Pembagian Tanggal",
        headers=["Partisi", "Tanggal Rekaman", "Jumlah Video", "Jumlah Frame", "Kondisi Lapangan"],
        rows=[
            ("Train", "03, 06, 09, 17 Nov 2026", "8 video", "67 frame", "Pagi, siang, dan sore bervariasi"),
            ("Validation", "10, 18 Nov 2026", "2 video", "18 frame", "Sore hujan basah dan pagi cerah"),
            ("Test", "04, 12 Nov 2026", "3 video", "26 frame", "Siang cerah, senja gelap, pagi cerah"),
            ("Total", "8 Tanggal Berbeda", "13 video", "111 frame", "Resolusi 960 × 540 piksel"),
        ],
        col_widths=[0.95, 1.55, 0.90, 0.95, 1.90],
        alignments=[
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.LEFT,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.LEFT,
        ],
        font_size=9.5,
    )

    add_image_with_caption(
        FIG1_PATH,
        "Gambar 1 Contoh Overlay Anotasi Ground Truth pada Empat Kondisi Pencahayaan Berbeda",
        width_in=6.05,
        space_after=8,
    )

    add_h2("2.2 Rekayasa Fitur Pengolahan Citra Digital dan Mitigasi Kondisi Senja")
    add_para(
        "Pendekatan awal versi pertama (v1) mengekstraksi 22 fitur piksel yang menggabungkan ruang warna HSV, CIE Lab, tekstur lokal berbasis deviasi standar, serta deteksi tepi Sobel dan Canny dari 536.000 sampel piksel latih. Meskipun versi pertama langsung melampaui baseline Naive Bayes secara signifikan pada kondisi siang hari, evaluasi rinci memperlihatkan penurunan tajam pada video senja tanggal 4 November dengan mIoU hanya berkisar 0.6698 hingga 0.6845. Pengukuran distribusi intensitas menyingkap penyebab fisik pergeseran tersebut, yaitu nilai median kanal kecerahan V pada lantai dan motor turun drastis dari 172.6 dan 104.0 pada data latih siang hari menjadi 95.7 dan 58.0 saat senja, disertai penyempitan kanal kromatik b pada ruang warna CIE Lab menuju titik netral 128 dari 131.4 dan 124.1 menjadi 129.0 dan 127.0."
    )
    add_para(
        "Upaya perbaikan awal dengan menormalisasi kecerahan terhadap median wilayah pengamatan justru memperburuk performa senja hingga mIoU 0.5656 karena muncul bias kepadatan parkir. Pada data latih, sekitar 65% wilayah pengamatan tertutup sepeda motor sehingga median intensitas wilayah bernilai gelap, sedangkan pada rekaman senja sekitar 70% area merupakan lantai kosong sehingga median wilayah bergeser jauh lebih terang dan mendistorsi skor normalisasi. Masalah ini diselesaikan pada iterasi kedua (v2) dengan menghapus seluruh fitur berbasis median global dan fitur kecerahan mentah L, lalu menggantinya dengan 26 fitur lokal yang murni invarian terhadap tingkat keterisian lahan. Kumpulan fitur tersebut mencakup representasi sudut warna H_shifted, H_sin, H_cos, saturasi S, kanal kecerahan V hasil peningkatan kontras adaptif CLAHE dengan clipLimit 2.5, kanal warna CIE Lab a dan b, kromatisitas RGB ternormalisasi r_norm, g_norm, b_norm, selisih kanal r_minus_b, pencahayaan Multi-Scale Retinex pada kanal V, kontras lokal jendela 31 × 31 piksel V_rel_31 dan b_rel_31, deviasi standar lokal pada skala 5, 11, dan 25 piksel termasuk pada kanal a, b, dan S, gradien tepi Sobel skala 5 dan 15, kepadatan tepi Canny skala 11 dan 21, respons Laplacian skala 11, serta rentang intensitas lokal jendela 15 piksel."
    )
    add_para(
        "Untuk menjembatani ketiadaan sampel rekaman senja murni pada data latih tanpa menyentuh data uji, diterapkan augmentasi senja sintetis berbasis fisika citra pada partisi latih sehingga jumlah sampel meningkat menjadi 603.000 piksel. Setiap frame latih yang diaugmentasi mengalami penskalaan gamma gelap acak yang dikombinasikan dengan peredaman kromatik menuju abu-abu netral dan sedikit derau sensor kamera, meniru karakteristik optik penurunan kecerahan serta desaturasi kanal b yang terjadi pada rekaman senja sebenarnya."
    )

    add_h2("2.3 Pemodelan Random Forest, XGBoost, dan Tuning Optuna")
    add_para(
        "Dua keluarga algoritma ensemble berbasis pohon keputusan dibandingkan secara langsung pada matriks fitur yang sama, yaitu Random Forest yang mengandalkan bagging paralel pohon dalam untuk menekan varians dan XGBoost yang membangun pohon secara sekuensial berbasis gradien untuk meminimalkan kesalahan residual secara ringkas. Pada iterasi v1 dan v2, kedua model dilatih dengan konfigurasi manual yang konservatif terhadap overfitting, yakni Random Forest dengan 250 pohon dan kedalaman terkendali serta XGBoost dengan kedalaman maksimum 7, min_child_weight 15, dan regularisasi L1 serta L2."
    )
    add_para(
        "Pada iterasi ketiga (v3), dilakukan pencarian hiperparameter otomatis menggunakan Optuna berbasis Tree-structured Parzen Estimator secara lokal dan bebas kebocoran data dengan 35 percobaan untuk XGBoost dan 18 percobaan untuk Random Forest. Fungsi objektif optimasi dihitung murni pada data validasi sebagai rata-rata berimbang antara mIoU pada frame validasi asli dan mIoU pada versi validasi yang diredupkan secara sintetis agar model yang terpilih tidak hanya unggul pada cahaya terang. Setelah model terbaik ditetapkan, parameter pascapemrosesan berupa sigma Gaussian Blur pada peta probabilitas, ambang batas keputusan probabilitas, serta ukuran kernel elips asimetris untuk Morphological Opening dan Morphological Closing dituning secara terpisah menggunakan data validasi untuk membersihkan bercak derau kecil sekaligus menutup celah sempit di badan sepeda motor."
    )

    add_h2("2.4 Proyeksi Bird's-Eye View dan Estimasi Kapasitas Ruang Kosong")
    add_para(
        "Evaluasi akhir tidak berhenti pada metrik segmentasi piksel, melainkan diterjemahkan langsung ke estimasi jumlah ruang parkir sepeda motor yang tersedia menggunakan algoritma proyeksi Bird's-Eye View (BEV) resmi repositori tim. Setiap frame uji disejajarkan terlebih dahulu terhadap frame referensi menggunakan pencocokan fitur SIFT dan estimasi homografi RANSAC dengan jumlah inliers minimal 619 titik agar pergeseran kecil kamera antarhari terkoreksi secara presisi. Masker prediksi lantai dan motor kemudian diproyeksikan melalui transformasi perspektif ke bidang atas berukuran 600 × 600 piksel menggunakan poligon wilayah pengamatan dari berkas JSON referensi."
    )
    add_para(
        "Pada bidang Bird's-Eye View yang telah bebas distorsi perspektif, komponen lantai kosong yang saling terhubung dihitung kapasitasnya berdasarkan rasio dimensi ruang satu sepeda motor dengan faktor pengisian FIT sebesar 0.7. Nilai estimasi kapasitas dari setiap model dibandingkan terhadap jumlah ruang kosong ground truth pada 26 frame uji menggunakan metrik Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE) dalam satuan jumlah sepeda motor."
    )

    # =========================================================================
    # BAB 3: HASIL PENGUJIAN DAN PEMBAHASAN
    # =========================================================================
    add_h1("3. Hasil Pengujian dan Pembahasan")

    add_h2("3.1 Perbandingan Performa Model Keseluruhan")
    add_para(
        "Seluruh model dari iterasi pertama hingga ketiga dievaluasi pada 26 frame uji yang sama sekali tidak pernah dilihat selama pelatihan, augmentasi, maupun pemilihan hiperparameter. Rekapitulasi lengkap perbandingan performa segmentasi, ketahanan kondisi senja, akurasi estimasi kapasitas ruang kosong pada Bird's-Eye View, waktu latih, kecepatan inferensi, dan ukuran berkas model disajikan pada Tabel 3.1."
    )

    add_bw_table(
        caption="Tabel 3.1 Perbandingan Performa Model pada 26 Frame Uji Held-Out",
        headers=[
            "Model",
            "mIoU",
            "IoU Lantai",
            "IoU Motor",
            "Macro-F1",
            "Pixel Acc",
            "Dusk mIoU",
            "MAE BEV",
            "RMSE BEV",
            "Train (s)",
            "Infer (ms)",
            "Size (MB)",
        ],
        rows=[
            ("Naive Bayes (Repo)", "0.6470", "0.5370", "0.7570", "−", "0.8110", "−", "12.81", "−", "−", "−", "−"),
            ("RF v1 (22 feat)", "0.7879", "0.7599", "0.8159", "0.8811", "0.8837", "0.6845", "3.23", "4.57", "103.98", "1467.9", "249.57"),
            ("XGB v1 (22 feat)", "0.7862", "0.7568", "0.8156", "0.8800", "0.8828", "0.6698", "4.08", "5.57", "10.09", "365.8", "1.45"),
            ("RF v2 (26 feat)", "0.7991", "0.7750", "0.8232", "0.8881", "0.8901", "0.7249", "2.31", "2.94", "152.23", "1704.5", "295.01"),
            ("XGB v2 (26 feat)", "0.8109", "0.7813", "0.8405", "0.8953", "0.8984", "0.6992", "3.62", "5.43", "38.55", "931.1", "1.98"),
            ("RF v3 (Optuna)", "0.8107", "0.7842", "0.8372", "0.8952", "0.8977", "0.7146", "2.77", "3.63", "513.05", "3093.3", "243.49"),
            ("XGB v3 (Optuna)", "0.8046", "0.7729", "0.8363", "0.8914", "0.8948", "0.6984", "4.46", "5.97", "53.93", "1897.6", "6.10"),
        ],
        col_widths=[1.10, 0.45, 0.47, 0.46, 0.48, 0.46, 0.47, 0.43, 0.45, 0.49, 0.51, 0.48],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.RIGHT] * 11,
        font_size=7.5,
    )

    add_para(
        "Hasil pengujian pada Tabel 3.1 menunjukkan lonjakan performa yang sangat besar dibandingkan baseline Naive Bayes repositori tim. Pada iterasi pertama sekalipun, Random Forest v1 dan XGBoost v1 langsung menaikkan mIoU dari 0.6470 menjadi 0.7879 dan 0.7862 atau meningkat sekitar 14 poin persentase, sekaligus memangkas kesalahan estimasi kapasitas parkir dari 12.81 motor turun menjadi 3.23 dan 4.08 motor. Penyebab utama lompatan tersebut adalah kemampuan pohon keputusan dalam mempartisi kombinasi fitur warna dan tekstur secara non-linear sehingga lantai yang tertutup bayangan tidak lagi tercampur dengan kelas sepeda motor."
    )
    add_para(
        "Secara keseluruhan, tidak ada satu model tunggal yang mendominasi seluruh aspek secara mutlak karena terdapat trade-off yang jelas antara akurasi segmentasi piksel, ketahanan senja, dan efisiensi komputasi. XGBoost v2 mencapai mIoU tertinggi sebesar 0.8109, IoU Motor tertinggi 0.8405, Macro-F1 tertinggi 0.8953, dan Pixel Accuracy tertinggi 0.8984 dengan ukuran berkas sangat ringkas hanya 1.98 MB serta waktu inferensi 931.1 ms per frame. Di sisi lain, Random Forest v2 meraih ketahanan senja tertinggi dengan Dusk mIoU 0.7249 sekaligus menghasilkan estimasi kapasitas ruang parkir paling akurat dengan MAE 2.31 motor dan RMSE 2.94 motor, walaupun memerlukan penyimpanan model sebesar 295.01 MB dan waktu inferensi 1704.5 ms per frame. Sementara itu, Random Forest v3 hasil tuning Optuna menyamai capaian mIoU puncak pada 0.8107 dan mencetak IoU Lantai tertinggi sebesar 0.7842."
    )

    add_h2("3.2 Analisis Fitur Dominan dan Efek Perbaikan Senja")
    add_para(
        "Transisi dari 22 fitur pada v1 menuju 26 fitur lokal invarian kepadatan pada v2 yang dipadukan dengan augmentasi senja sintetis memberikan peningkatan merata pada kedua arsitektur. Random Forest meningkat dari mIoU 0.7879 menjadi 0.7991 dengan lonjakan Dusk mIoU sebesar 4.04 poin dari 0.6845 ke 0.7249, sedangkan XGBoost naik dari 0.7862 menjadi 0.8109 dengan kenaikan Dusk mIoU dari 0.6698 ke 0.6992. Distribusi kontribusi 20 fitur paling berpengaruh bagi kedua model v2 diperlihatkan pada Gambar 2."
    )

    add_image_with_caption(
        FIG2_PATH,
        "Gambar 2 Peringkat 20 Fitur Pengolahan Citra Digital Paling Berpengaruh pada Model v2",
        width_in=6.05,
        space_after=8,
    )

    add_para(
        "Grafik kepentingan fitur pada Gambar 2 memperlihatkan bahwa fitur kromatik lokal dan variansi tekstur memegang kendali utama pada pemisahan lantai dan motor. Fitur selisih kromatik r_minus_b menempati peringkat pertama pada Random Forest v2 maupun XGBoost v2, diikuti secara konsisten oleh deviasi saturasi lokal std_11_S, kromatisitas merah ternormalisasi r_norm, komponen arah warna H_cos, kanal warna CIE Lab b_lab, deviasi intensitas jendela 25 piksel std_25, pencahayaan Multi-Scale Retinex msr_v, dan gradien tepi Sobel skala 15. Pola ini selaras dengan karakteristik fisik objek di lapangan karena permukaan paving atau aspal lantai memiliki warna hangat seragam dengan variasi saturasi dan tepi rendah meskipun gelap terkena bayangan, sedangkan kumpulan sepeda motor memunculkan lonjakan variasi saturasi, pantulan dingin keabuan atau metalik, serta gradien tepi yang tajam. Perbandingan visual hasil segmentasi model v2 lintas kondisi siang, senja, dan pagi ditampilkan pada Gambar 3."
    )

    add_image_with_caption(
        FIG3_PATH,
        "Gambar 3 Perbandingan Visual Segmentasi Random Forest v2 dan XGBoost v2 pada Citra Uji",
        width_in=6.05,
        space_after=8,
    )

    add_h2("3.3 Evaluasi Tuning Optuna dan Analisis Generalisasi")
    add_para(
        "Optimasi hiperparameter menggunakan Optuna dijalankan terpisah dari data uji untuk menguji apakah konfigurasi pohon yang lebih kompleks mampu menembus batas mIoU 0.81. Rangkuman konfigurasi terbaik hasil pencarian 18 percobaan Random Forest dan 35 percobaan XGBoost beserta parameter pascapemrosesan optimal pada data validasi tercantum pada Tabel 3.2, sedangkan kurva konvergensi percobaan Optuna diperlihatkan pada Gambar 4."
    )

    add_bw_table(
        caption="Tabel 3.2 Hiperparameter dan Pascapemrosesan Terbaik Hasil Pencarian Optuna",
        headers=["Aspek Konfigurasi", "Random Forest v3 (18 Trial)", "XGBoost v3 (35 Trial)"],
        rows=[
            ("Objektif Validasi Terbaik", "Balanced mIoU 0.8420", "Balanced mIoU 0.8539"),
            ("Skor Validasi Asli / Senja", "Orig 0.8683 / Dusk 0.8411", "Orig 0.8745 / Dusk 0.8602"),
            ("Validasi BEV MAE", "1.89 ruang kosong", "1.89 ruang kosong"),
            ("Struktur Pohon Utama", "n_estimators 300, max_depth 27, entropy", "best_iter 812, max_depth 9, lr 0.0459"),
            ("Regularisasi dan Sampling", "min_leaf 4, min_split 17, max_feat 0.540", "min_child 6.85, sub 0.892, col 0.607"),
            ("Penalti Tambahan", "max_samples 0.809", "alpha 2.59, lambda 2.89, gamma 0.379"),
            ("Pascapemrosesan Morfologi", "blur 2.5, thr 0.42, open 19, close 11", "blur 2.5, thr 0.42, open 15, close 11"),
        ],
        col_widths=[1.85, 2.20, 2.20],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
        font_size=9.0,
    )

    add_image_with_caption(
        FIG4_PATH,
        "Gambar 4 Riwayat Konvergensi Pencarian Hiperparameter Optuna pada Data Validasi",
        width_in=6.05,
        space_after=8,
    )

    add_para(
        "Perbandingan antara skor validasi pada Tabel 3.2 dan skor uji pada Tabel 3.1 memberikan temuan penting mengenai perilaku generalisasi kedua algoritma. Pada Random Forest, pencarian Optuna berhasil mengangkat mIoU uji sebesar 1.16 poin dari 0.7991 menjadi 0.8107 serta memperbaiki IoU Lantai ke nilai tertinggi 0.7842 karena mekanisme bagging secara alami meredam varians pohon dalam saat max_depth diperbesar ke 27 dengan batas min_samples_split 17. Sebaliknya, pada XGBoost v3 terjadi penurunan kecil pada data uji dari 0.8109 menjadi 0.8046 walaupun skor validasinya naik hingga 0.8539. Fenomena tersebut merupakan gejala overfitting ringan terhadap 2 hari validasi karena Optuna memilih kedalaman pohon lebih agresif yaitu max_depth 9 dengan min_child_weight lebih longgar 6.85 dan 812 iterasi dibandingkan konfigurasi konservatif XGBoost v2 yang membatasi kedalaman pada 7 dan min_child_weight pada 15."
    )
    add_para(
        "Selain faktor kapasitas model, stagnasi skor keseluruhan di sekitar kisaran 0.8107 sampai 0.8109 mengindikasikan bahwa sistem berbasis fitur lokal piksel mulai menyentuh batas informasi visual lapangan. Pemeriksaan galat kualitatif pada Gambar 5 menunjukkan bahwa sumber kesalahan paling persisten pada kondisi senja berasal dari kisi saluran drainase berbahan logam gelap yang melintang di tengah lantai parkir. Secara tekstur dan intensitas lokal, kisi logam gelap tersebut memiliki gradien tepi tinggi dan pantulan gelap yang hampir identik dengan roda atau bagian bawah sepeda motor sehingga model berbasis jendela piksel terbatas sulit membedakannya tanpa konteks bentuk global kendaraan."
    )

    add_image_with_caption(
        FIG5_PATH,
        "Gambar 5 Perbandingan Visual Segmentasi Random Forest v3 dan XGBoost v3 Hasil Tuning Optuna",
        width_in=6.05,
        space_after=8,
    )

    add_h2("3.4 Evaluasi Kapasitas Ruang Kosong Berbasis Bird's-Eye View")
    add_para(
        "Analisis per video uji dilakukan untuk melihat bagaimana kesalahan segmentasi pada tiap kondisi cuaca dan waktu diterjemahkan menjadi selisih jumlah ruang parkir kosong pada proyeksi Bird's-Eye View. Rincian metrik mIoU, rata-rata prediksi ruang kosong, dan MAE per video uji untuk seluruh model ditampilkan pada Tabel 3.3, sedangkan visualisasi langsung pemetaan slot kosong pada bidang Bird's-Eye View ditunjukkan pada Gambar 6."
    )

    add_bw_table(
        caption="Tabel 3.3 Rincian Performa mIoU dan Estimasi Kapasitas BEV per Video Uji",
        headers=[
            "Model",
            "Siang 03 (9 fr, GT 38.3)\nmIoU / Pred / MAE",
            "Senja 04 (9 fr, GT 23.8)\nmIoU / Pred / MAE",
            "Pagi 09 (8 fr, GT 17.5)\nmIoU / Pred / MAE",
        ],
        rows=[
            ("RF v1 (22 feat)", "0.8109 / − / 1.44", "0.6845 / − / 7.33", "0.8378 / − / 0.62"),
            ("XGB v1 (22 feat)", "0.8133 / − / 2.22", "0.6698 / − / 8.89", "0.8441 / − / 0.75"),
            ("RF v2 (26 feat)", "0.8062 / 39.2 / 1.56", "0.7249 / 19.4 / 4.33", "0.8391 / 17.6 / 0.88"),
            ("XGB v2 (26 feat)", "0.8450 / 38.2 / 0.56", "0.6992 / 14.9 / 8.89", "0.8572 / 16.4 / 1.12"),
            ("RF v3 (Optuna)", "0.8323 / 38.2 / 1.22", "0.7146 / 18.1 / 5.67", "0.8554 / 16.2 / 1.25"),
            ("XGB v3 (Optuna)", "0.8268 / 36.8 / 2.00", "0.6984 / 14.2 / 9.56", "0.8603 / 16.0 / 1.50"),
        ],
        col_widths=[1.25, 1.67, 1.67, 1.66],
        alignments=[
            WD_ALIGN_PARAGRAPH.LEFT,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.CENTER,
        ],
        font_size=8.8,
    )

    add_image_with_caption(
        FIG6_PATH,
        "Gambar 6 Visualisasi Estimasi Kapasitas Ruang Parkir Kosong pada Proyeksi Bird's-Eye View",
        width_in=6.05,
        space_after=8,
    )

    add_para(
        "Rincian pada Tabel 3.3 menegaskan bahwa performa sistem pada kondisi terang telah mencapai tingkat akurasi operasional yang sangat tinggi. Pada video siang hari dengan kapasitas ground truth rata-rata 38.3 ruang kosong, XGBoost v2 memprediksi rata-rata 38.2 ruang dengan MAE hanya 0.56 motor dan mIoU 0.8450. Pada video pagi hari dengan ground truth 17.5 ruang, seluruh model v2 dan v3 mencapai mIoU antara 0.8391 hingga 0.8603 dengan kesalahan kapasitas sekitar 0.88 hingga 1.50 motor, menunjukkan bahwa algoritma penghitungan kapasitas Bird's-Eye View bekerja sangat presisi ketika kontras permukaan lantai terjaga."
    )
    add_para(
        "Sebaliknya, hampir seluruh kontribusi kesalahan agregat berasal dari video senja 04-04-11 dengan kapasitas ground truth rata-rata 23.8 ruang kosong. Pada kondisi redup tersebut, seluruh model cenderung melakukan prediksi lebih rendah dari kenyataan karena sebagian area lantai gelap dan jalur kisi drainase terklasifikasi sebagai motor, sehingga area lantai yang seharusnya utuh pada bidang Bird's-Eye View terbelah menjadi fragmen-fragmen kecil yang tidak lagi memenuhi ukuran satu ruang motor. Random Forest v2 menunjukkan ketahanan paling baik pada kondisi ekstrem ini dengan prediksi rata-rata 19.4 ruang dan MAE 4.33 motor, jauh lebih stabil dibandingkan XGBoost v2 dengan MAE 8.89 motor maupun XGBoost v3 dengan MAE 9.56 motor. Oleh karena itu, dalam pemilihan model implementasi, Random Forest v2 unggul apabila prioritas utama adalah keandalan hitung ruang parkir di segala cuaca termasuk senja, sedangkan XGBoost v2 lebih ideal apabila prioritas diarahkan pada efisiensi memori sebesar 1.98 MB, kecepatan inferensi hampir dua kali lebih cepat, serta akurasi maksimal pada jam operasional siang hari."
    )

    # =========================================================================
    # BAB 4: KESIMPULAN
    # =========================================================================
    add_h1("4. Kesimpulan")
    add_para(
        "Pembangunan pipeline deteksi area parkir kosong berbasis 26 fitur pengolahan citra digital lokal dan augmentasi senja sintetis pada dataset 111 frame bebas kebocoran data berhasil melampaui baseline Naive Bayes repositori tim secara signifikan. Nilai mIoU meningkat dari 0.6470 menjadi 0.8109 pada XGBoost v2 dan 0.8107 pada Random Forest v3 hasil tuning Optuna, sementara kesalahan estimasi kapasitas ruang kosong pada proyeksi Bird's-Eye View turun drastis dari 12.81 motor menjadi hanya 2.31 motor pada Random Forest v2. Penghapusan fitur normalisasi berbasis median wilayah terbukti menjadi langkah kunci untuk menghilangkan bias tingkat keterisian lahan sehingga performa pada kondisi senja meningkat stabil tanpa mengorbankan akurasi siang dan pagi hari."
    )
    add_para(
        "Perbandingan antara kedua algoritma ensemble memperlihatkan keunggulan pelengkap yang jelas, di mana XGBoost v2 menawarkan keseimbangan terbaik antara mIoU tertinggi 0.8109, kecepatan inferensi 931.1 ms per frame, dan ukuran model ringkas 1.98 MB, sedangkan Random Forest v2 memberikan estimasi kapasitas ruang kosong paling akurat dengan MAE 2.31 motor dan ketahanan terbaik pada pencahayaan senja dengan Dusk mIoU 0.7249. Tuning Optuna berhasil mengangkat performa Random Forest sebesar 1.16 poin mIoU, namun pada XGBoost cenderung sedikit overfit terhadap dua hari validasi karena kedalaman pohon yang lebih tinggi. Untuk menembus batas akurasi di atas 0.81 mIoU sekaligus mengatasi kesalahan lokal pada kisi drainase logam saat senja, pengembangan lanjutan dapat diarahkan pada penambahan rekaman latih kondisi senja sebenarnya atau penerapan model segmentasi berbasis konteks spasial global seperti YOLO11n-seg atau SegFormer."
    )

    # =========================================================================
    # DAFTAR PUSTAKA
    # =========================================================================
    add_h1("Daftar Pustaka")
    references = [
        "1. Breiman, L., Random Forests, Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.",
        "2. Chen, T., dan Guestrin, C., XGBoost: A Scalable Tree Boosting System, Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785-794, 2016.",
        "3. Lowe, D. G., Distinctive Image Features from Scale-Invariant Keypoints, International Journal of Computer Vision, vol. 60, no. 2, pp. 91-110, 2004.",
        "4. Jobson, D. J., Rahman, Z., dan Woodell, G. A., A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes, IEEE Transactions on Image Processing, vol. 6, no. 7, pp. 965-976, 1997.",
        "5. Akiba, T., Sano, S., Yanase, T., Ohta, T., dan Koyama, M., Optuna: A Next-generation Hyperparameter Optimization Framework, Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 2623-2631, 2019.",
        "6. Bradski, G., The OpenCV Library, Dr. Dobb's Journal of Software Tools, 2000.",
    ]
    for ref in references:
        add_para(ref, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=11, space_before=0, space_after=3, line_spacing=1.15)

    doc.save(str(OUT_DOCX))
    print(f"[+] DOCX berhasil disimpan: {OUT_DOCX}")


def export_pdf_and_proofs():
    word = None
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc_obj = word.Documents.Open(str(OUT_DOCX.resolve()))
        doc_obj.SaveAs(str(OUT_PDF.resolve()), FileFormat=17)
        doc_obj.Close(False)
        print(f"[+] PDF berhasil diekspor: {OUT_PDF}")
    finally:
        if word is not None:
            word.Quit()

    if PROOFS_DIR.exists():
        for old_png in PROOFS_DIR.glob("proof_page_*.png"):
            old_png.unlink()
    PROOFS_DIR.mkdir(parents=True, exist_ok=True)

    pdf_doc = fitz.open(str(OUT_PDF))
    page_count = len(pdf_doc)
    zoom = 150.0 / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for idx in range(page_count):
        page = pdf_doc.load_page(idx)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_png = PROOFS_DIR / f"proof_page_{idx + 1}.png"
        pix.save(str(out_png))
    pdf_doc.close()
    print(f"[+] Total halaman PDF: {page_count} | Proof PNG tersimpan di: {PROOFS_DIR}")


if __name__ == "__main__":
    build_report()
    export_pdf_and_proofs()
