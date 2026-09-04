import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
import string


"""EMBROIDERY_SYMBOLS = [
    "🟆", "🞴", "🞺", "🞬", "🞔", "🞅", "🞚", "🞤", "●", "▲", "■", "✜", "★", "▼", "◄", "►", "★", "🛆", "🛇", "𞢹", "#", "&", "%", "$", "@", "?", "!", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", 
    "L", "M", "N", "P", "R", "S", "T", "U", "V", "W", "Y", "Z"
]"""

SYMBOL_POOL = list(string.digits[1:] + string.digits[:1]) + list(string.ascii_uppercase) + list(string.ascii_lowercase) + list("!@#$%&*+=?")

def calculate_optimal_cell_size(pattern_width, pattern_height, margin_mm):
    page_width_mm = A4[0] / mm
    page_height_mm = A4[1] / mm

    max_cell_by_width = (page_width_mm - 2 * margin_mm) / pattern_width
    max_cell_by_height = (page_height_mm - 2 * margin_mm) / pattern_height

    if max_cell_by_width > max_cell_by_height:
        max_cell_size = max_cell_by_height
    else:
        max_cell_size = max_cell_by_width

    if max_cell_size < 3:
        return 3
    if max_cell_size > 10:
        return 10

    return max_cell_size

def assign_symbols(codes: list[str]) -> dict[str, str]:
    unique_codes = sorted(set(codes))
    if len(unique_codes) > len(SYMBOL_POOL):
        raise ValueError(f"Too many colors ({len(unique_codes)})for symbol pool ({len(SYMBOL_POOL)})")
    return {code: SYMBOL_POOL[i] for i, code in enumerate(unique_codes)}

def get_symbol_color(lightness: float, threshold: float = 55.0) -> colors.Color:
    return colors.HexColor("#000000") if lightness > threshold else colors.HexColor("#FFFFFF")


def export_pattern_to_pdf(
    grid_codes: np.ndarray,
    dmc_df: pd.DataFrame,
    output_path: str,
) -> None:

    n_rows, n_cols = grid_codes.shape

    if "L" not in dmc_df.columns:
        from src.colors.dmc import rgb_to_lab
        dmc_df = rgb_to_lab(dmc_df)

    code_to_hex = dict(zip(dmc_df["code"], dmc_df["hex"]))
    code_to_lightness = dict(zip(dmc_df["code"], dmc_df["L"]))
    code_to_symbol = assign_symbols(grid_codes.flatten().tolist())

    page_width, page_height = A4
    c = canvas.Canvas(output_path, pagesize=A4)

    margin_mm = 15
    margin = margin_mm * mm
    cell = calculate_optimal_cell_size(max_cells_per_page, max_cells_per_page, margin_mm) * mm

    max_cells_per_page = 60
    num_pages_x = max(1, int(np.ceil(n_cols / max_cells_per_page)))
    num_pages_y = max(1, int(np.ceil(n_rows / max_cells_per_page)))

    cols_per_page = int(np.ceil(n_cols / num_pages_x))
    rows_per_page = int(np.ceil(n_rows / num_pages_y))

    total_pattern_pages = num_pages_x * num_pages_y
   
    printable_width = page_width - 2 * margin
    printable_height = page_height - 2 * margin

    for page_y in range(num_pages_y):
        row_start = page_y * rows_per_page
        row_end = min(row_start + rows_per_page, n_rows)
        cells_this_page_rows = row_end - row_start

        for page_x in range(num_pages_x):
            col_start = page_x * cols_per_page
            col_end = min(col_start + cols_per_page, n_cols)
            cells_this_page_cols = col_end - col_start

            # how much space will the pattern take on specific page
            pattern_width_this_page = cells_this_page_cols * cell
            pattern_height_this_page = cells_this_page_rows * cell

            # offset to draw pattern in the middle of the page
            x_offset = margin + (printable_width - pattern_width_this_page) / 2
            y_offset = margin + (printable_height - pattern_height_this_page) / 2

            # !!! DRAWING !!!

            # draw cells with symbol - white for dark colors, black for bright colors for contrast
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    code = grid_codes[row, col]
                    hex_color = code_to_hex.get(code, "#FFFFFF")

                    local_col = col - col_start
                    local_row = row - row_start

                    x = x_offset + local_col * cell
                    y = page_height - y_offset - (local_row + 1) * cell

                    c.setFillColor(hex_color)
                    c.rect(x, y, cell, cell, fill=1, stroke=1)

                    lightness = code_to_lightness.get(code, 100)
                    symbol = code_to_symbol[code]
                    symbol_font_size = cell * 0.8

                    c.setFillColor(get_symbol_color(lightness))
                    c.setFont("Helvetica-Bold", symbol_font_size)
                    c.drawCentredString(x + cell / 2, y + cell * 0.28, symbol)

            # draw bold lines every 10 rows
            c.setStrokeColor(colors.HexColor("#000000"))
            for row in range(row_start, row_end + 1):
                local_row = row - row_start
                y_line = page_height - y_offset - local_row * cell
                
                if row > 0 and row % 10 == 0:
                    c.setLineWidth(2.0)
                else:
                    c.setLineWidth(0.3)
                
                c.line(x_offset, y_line, x_offset + cells_this_page_cols * cell, y_line)

            # draw bold lines every 10 columns
            for col in range(col_start, col_end + 1):
                local_col = col - col_start
                x_line = x_offset + local_col * cell
                
                if col > 0 and col % 10 == 0:
                    c.setLineWidth(2.0)
                else:
                    c.setLineWidth(0.3)
                    
                c.line(x_line, y_offset, x_line, y_offset + cells_this_page_rows * cell)

            # enumerate every 10th cell in a row
            c.setFillColor(colors.HexColor("#000000"))
            c.setFont("Helvetica", 8)
            
            for row in range(row_start, row_end):
                if (row + 1) % 10 == 0:
                    local_row = row - row_start
                    x_text = x_offset - 5
                    y_text = page_height - y_offset - (local_row + 1) * cell + (cell / 4)
                    c.drawRightString(x_text, y_text, str(row + 1))

            # enumerate every 10th cell in a column
            for col in range(col_start, col_end):
                if (col + 1) % 10 == 0:
                    local_col = col - col_start
                    x_text = x_offset + local_col * cell + (cell / 2)
                    y_text = page_height - y_offset + 3
                    c.drawCentredString(x_text, y_text, str(col + 1))

            # header
            current_page_num = page_y * num_pages_x + page_x + 1
            
            header_line_1 = f"Page {current_page_num}/{total_pattern_pages} - Section ({page_x + 1}, {page_y + 1})"
            header_line_2 = f"Columns {col_start}-{col_end} | Rows {row_start}-{row_end} | Size: {cells_this_page_cols}x{cells_this_page_rows}"

            c.setFillColor(colors.HexColor("#000000"))
            
            c.setFont("Helvetica-Bold", 20)
            c.drawCentredString(page_width / 2, page_height - 12 * mm, header_line_1)
            
            c.setFont("Helvetica", 14)
            c.drawCentredString(page_width / 2, page_height - 20 * mm, header_line_2)

            # page number
            current_page_num = page_y * num_pages_x + page_x + 1
            
            c.setFont("Helvetica", 9)
            
            page_center_x = page_width / 2
            footer_y = 15 * mm
            
            page_text = f"Page {current_page_num} of {total_pattern_pages}"
            c.drawCentredString(page_center_x, footer_y, page_text)

            c.showPage()

    c.save()


def main():
    from src.colors.dmc import load_dmc_palette, rgb_to_lab, build_lab_tree
    from src.processing.pipeline import image_to_dmc_pattern

    dmc_df = load_dmc_palette()
    dmc_df = rgb_to_lab(dmc_df)
    dmc_tree = build_lab_tree(dmc_df)

    test_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    test_img[:500, :500] = [255, 0, 0]
    test_img[:500, 500:] = [0, 255, 0]
    test_img[500:, :500] = [0, 0, 255]
    test_img[500:, 500:] = [255, 255, 0]

    grid, _ = image_to_dmc_pattern(
        test_img, target_width=100, target_height=150, n_colors=4,
        dmc_df=dmc_df, dmc_tree=dmc_tree,
    )

    export_pattern_to_pdf(grid, dmc_df, "test_output_large.pdf")
    print("grid shape:", grid.shape)
    print("PDF saved as test_output_large.pdf")


if __name__ == "__main__":
    main()