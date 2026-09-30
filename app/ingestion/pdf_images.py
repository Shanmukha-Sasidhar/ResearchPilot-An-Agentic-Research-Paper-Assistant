# pyrefly: ignore [missing-import]
import fitz
import os


def render_page(
    pdf_path: str,
    page_number: int,
    output_dir: str = "data/processed/pages"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    pdf = fitz.open(pdf_path)

    page_index = page_number - 1

    page = pdf[page_index]

    matrix = fitz.Matrix(
        2,
        2
    )

    pixmap = page.get_pixmap(
        matrix=matrix
    )

    image_path = os.path.join(
        output_dir,
        f"page_{page_number}.png"
    )

    pixmap.save(
        image_path
    )

    pdf.close()

    return image_path