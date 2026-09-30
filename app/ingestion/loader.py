from pathlib import Path

# pyrefly: ignore [missing-import]
import fitz


def load_pdf(file_path: str):

    document = fitz.open(file_path)

    return document