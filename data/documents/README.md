# Documents Directory

This directory is for storing medical documents that will be indexed by the RAG system.

## Directory Structure

Organize your documents by file type:

- `txt/` - Plain text files (.txt)
- `pdf/` - PDF documents (.pdf)
- `csv/` - CSV data files (.csv)
- `md/` - Markdown files (.md)
- `excel/` - Excel spreadsheets (.xlsx, .xls)
- `word/` - Word documents (.docx)

## Supported Formats

The system supports the following file formats:
- `.txt` - Plain text medical documents, guidelines, FAQs
- `.pdf` - Medical research papers, clinical guidelines
- `.csv` - Structured medical data, drug information
- `.md` - Markdown formatted documentation
- `.xlsx` / `.xls` - Excel spreadsheets with medical data, drug databases, patient records
- `.docx` - Word documents with medical content, reports, guidelines

## Usage

1. Place your medical documents in the appropriate subdirectory based on file type
2. Run `python main.py` from the project root
3. On first run, the system will automatically:
   - Load all documents from this directory
   - Split them into chunks
   - Generate embeddings using ZhipuAI embedding-3
   - Build a FAISS index
   - Save the index to `data/indexes/zhipu/`

## Notes

- Files are processed recursively from subdirectories
- Hidden files (starting with `.`) are excluded
- Ensure documents are UTF-8 encoded for best results
- Large documents will be automatically split into manageable chunks
- The system preserves file metadata (path, name, type, etc.)

## Example Structure

```
data/documents/
├── txt/
│   ├── diabetes_overview.txt
│   ├── hypertension_guidelines.txt
│   └── common_medications.txt
├── pdf/
│   ├── clinical_guidelines_2024.pdf
│   └── research_paper_diabetes.pdf
├── csv/
│   └── drug_interactions.csv
├── md/
│   └── medical_faq.md
├── excel/
│   ├── drug_database.xlsx
│   └── patient_records.xls
└── word/
    ├── medical_report.docx
    └── clinical_guidelines.docx
```

## Index Rebuilding

To rebuild the index with new documents:
1. Add new documents to the appropriate subdirectories
2. Delete the existing index: `rm -rf data/indexes/zhipu/`
3. Run `python main.py` to rebuild the index

The system will automatically detect the missing index and rebuild it from all documents in this directory.

