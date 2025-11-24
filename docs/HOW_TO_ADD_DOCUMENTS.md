# How to Add Documents to MediMind Agent

## Complete Workflow

### 1. Place Your Document

Put your medical document in the appropriate directory based on file type:

```bash
# Windows - TXT file
copy your_medical_document.txt data\documents\txt\

# Windows - Excel file
copy drug_database.xlsx data\documents\excel\

# Linux/Mac - TXT file
cp your_medical_document.txt data/documents/txt/

# Linux/Mac - Excel file
cp drug_database.xlsx data/documents/excel/
```

**Example file structure:**
```
data/documents/
├── txt/
│   ├── diabetes_overview.txt
│   ├── hypertension_guide.txt
│   └── medication_reference.txt
├── excel/
│   ├── drug_database.xlsx
│   └── patient_records.xls
├── pdf/
│   └── clinical_guidelines.pdf
├── csv/
│   └── drug_interactions.csv
└── word/
    └── medical_report.docx     # Your new Word file
```

**Supported file types:**
- `.txt` - Plain text
- `.pdf` - PDF documents
- `.csv` - CSV data
- `.md` - Markdown
- `.xlsx` / `.xls` - Excel spreadsheets
- `.docx` - Word documents

### 2. Rebuild the Index

The agent needs to rebuild the index to include your new document.

**Option A: Using the rebuild script (Recommended)**

```bash
python scripts/rebuild_index.py
```

This script will:
- Delete the old index
- Load ALL documents (including your new one)
- Build a new index with embeddings
- Save it to `data/indexes/zhipu/`

**Option B: Manual rebuild**

```bash
# Windows PowerShell
Remove-Item -Recurse -Force data\indexes\zhipu
python main.py

# Linux/Mac
rm -rf data/indexes/zhipu
python main.py
```

### 3. Start the Agent

```bash
python main.py
```

The agent will now have access to your new document!

### 4. Verify the Document is Loaded

When you run `python scripts/rebuild_index.py`, you'll see output like:

```
[3/4] Loading documents from data/documents...
Loaded 16 documents

Documents loaded:
  1. diabetes_overview.txt (5234 chars)
  2. hypertension_guide.txt (4123 chars)
  3. medication_reference.txt (3456 chars)
  4. your_medical_document.txt (2890 chars)  ← Your new file!
  ... and 12 more documents
```

### 5. Test with a Question

```
User: [Ask a question related to your new document]
