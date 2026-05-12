from pypdf import PdfReader

reader = PdfReader('data/pdfs/general-guidelines-for-admissions-lpu.pdf')
page = reader.pages[0]
text = page.extract_text()
print('Text length:', len(text))
print('First 500 chars:')
print(repr(text[:500]))