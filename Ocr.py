import pytesseract
import re
from pdf2image import convert_from_path

LANG_DEV = "hin+mar"
LANG_ENG = "eng"
CONFIG = r"--oem 3 --psm 7"   # PSM 7 = single text line / word


def detect_script(text):
    if re.search(r'[\u0900-\u097F]', text):
        return "DEV"
    elif re.search(r'[A-Za-z]', text):
        return "ENG"
    return "OTHER"


def extract_from_scanned_pdf(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    final_text = ""

    for page_no, page in enumerate(pages, 1):
        print(f"OCR Page {page_no}/{len(pages)}")

        # STEP 1: get word-level boxes
        data = pytesseract.image_to_data(
            page,
            lang="hin+eng+mar",
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT
        )

        n = len(data["text"])
        words_out = []

        for i in range(n):
            word = data["text"][i].strip()
            if not word:
                continue

            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i]
            )

            crop = page.crop((x, y, x + w, y + h))

            script = detect_script(word)

            # STEP 2: re-OCR EACH WORD with correct language
            if script == "DEV":
                clean = pytesseract.image_to_string(
                    crop, lang=LANG_DEV, config=CONFIG
                )
            elif script == "ENG":
                clean = pytesseract.image_to_string(
                    crop, lang=LANG_ENG, config=CONFIG
                )
            else:
                clean = word

            clean = clean.strip()
            if clean:
                words_out.append(clean)

        final_text += " ".join(words_out) + "\n\n"

    return final_text
