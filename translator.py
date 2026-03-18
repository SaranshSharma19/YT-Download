import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from fpdf import FPDF
from googletrans import Translator
import os

# Function to extract text using OCR
def extract_text_with_ocr(pdf_file):
    # Save the uploaded file temporarily
    temp_file_path = "temp_uploaded_file.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    # Convert PDF to images and extract text
    images = convert_from_path(temp_file_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)

    # Clean up the temporary file
    os.remove(temp_file_path)
    return text

# Function to translate extracted text
def translate_extracted(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Function to create a PDF from translated text
def create_pdf(translated_text, output_path):
    fpdf = FPDF()
    fpdf.set_font("Helvetica", size=7)
    fpdf.add_page()
    fpdf.multi_cell(w=0, h=5, txt=translated_text.encode("latin-1", errors="replace").decode("latin-1"))
    fpdf.output(output_path)

# Main Streamlit app
def main():
    st.title("PDF Translator with OCR")
    st.write("Upload a PDF file and select a language to translate the text.")

    # Language options
    languages = {
        'English': 'en',
        'Spanish': 'es',
        'French': 'fr',
        'German': 'de',
        'Italian': 'it',
        'Portuguese': 'pt',
        'Chinese (Simplified)': 'zh-CN',
        'Japanese': 'ja',
        'Korean': 'ko',
        'Russian': 'ru',
        'Arabic': 'ar',
        'Hindi': 'hi',
        'Bengali': 'bn',
        'Turkish': 'tr',
        'Vietnamese': 'vi',
        'Thai': 'th',
        'Polish': 'pl',
        'Dutch': 'nl',
        'Swedish': 'sv',
        'Danish': 'da',
        'Finnish': 'fi',
        'Norwegian': 'no',
        'Czech': 'cs',
        'Hungarian': 'hu',
        'Romanian': 'ro',
        'Bulgarian': 'bg',
        'Ukrainian': 'uk',
        'Hebrew': 'iw',
        'Malay': 'ms',
        'Indonesian': 'id',
        'Filipino': 'tl',
        'Swahili': 'sw',
        'Serbian': 'sr',
        'Slovak': 'sk',
        'Catalan': 'ca',
        'Croatian': 'hr',
        'Lithuanian': 'lt',
        'Latvian': 'lv',
        'Estonian': 'et',
    }

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    selected_language = st.selectbox("Select Language", list(languages.keys()))

    if st.button("Translate"):
        if uploaded_file is not None:
            # Read the PDF and translate using OCR
            extracted_text = extract_text_with_ocr(uploaded_file)
            if extracted_text:
                translated_text = translate_extracted(extracted_text, languages[selected_language])
                output_path = "translated_output.pdf"
                create_pdf(translated_text, output_path)

                # Provide download link
                st.success("Translation completed!")
                st.download_button("Download Translated PDF", data=open(output_path, "rb"), file_name="translated_output.pdf")

if __name__ == "__main__":
    main()