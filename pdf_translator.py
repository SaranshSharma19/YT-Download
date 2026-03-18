import streamlit as st
import pdfplumber
from fpdf import FPDF
from googletrans import Translator

# Function to extract text from a PDF page
def extract(page):
    return page.extract_text()

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
    st.title("PDF Translator")
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
            # Read the PDF and translate
            with pdfplumber.open(uploaded_file) as pdf:
                translated_text = ""
                for page in pdf.pages:
                    extracted = extract(page)
                    if extracted:
                        translated = translate_extracted(extracted, languages[selected_language])
                        translated_text += translated + "\n\n"

            # Create and save the translated PDF
            output_path = "translated_output.pdf"
            create_pdf(translated_text, output_path)

            # Provide download link
            st.success("Translation completed!")
            st.download_button("Download Translated PDF", data=open(output_path, "rb"), file_name="translated_output.pdf")

if __name__ == "__main__":
    main()