# pyrefly: ignore [missing-import]
import pdfplumber


def parse_pdf(file_path: str):

    pages = []

    with pdfplumber.open(file_path) as pdf:

        for page_number, page in enumerate(pdf.pages):

            text = page.extract_text()

            if text and text.strip():

                pages.append(
                    {
                        "page_number": page_number + 1,
                        "text": text.strip(),
                    }
                )

    return pages