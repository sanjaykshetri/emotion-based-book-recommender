# Convert Markdown to PDF
# This script converts the executive presentation to a professional PDF

"""
Instructions for creating the PDF:

Option 1: Use Markdown to PDF tools
--------------------------------------
1. Install markdown-pdf:
   npm install -g markdown-pdf
   
2. Convert:
   markdown-pdf docs/Executive_Presentation.md -o docs/Executive_Presentation.pdf

Option 2: Use Pandoc (Recommended)
-----------------------------------
1. Install Pandoc: https://pandoc.org/installing.html

2. Install LaTeX (for PDF generation):
   - Windows: MiKTeX (https://miktex.org/)
   - Mac: MacTeX (https://tug.org/mactex/)
   
3. Convert with styling:
   pandoc docs/Executive_Presentation.md -o docs/Executive_Presentation.pdf \
   --pdf-engine=xelatex \
   -V geometry:margin=1in \
   -V fontsize=11pt \
   -V colorlinks=true

Option 3: Use VS Code Extension
--------------------------------
1. Install "Markdown PDF" extension in VS Code
2. Open docs/Executive_Presentation.md
3. Right-click → "Markdown PDF: Export (pdf)"

Option 4: Use Online Converter
-------------------------------
1. Visit: https://www.markdowntopdf.com/
2. Upload: docs/Executive_Presentation.md
3. Download PDF

Option 5: Use Python (Automated)
---------------------------------
"""

import subprocess
import sys
from pathlib import Path

def convert_to_pdf():
    """Convert markdown to PDF using available tools."""
    
    md_file = Path("docs/Executive_Presentation.md")
    pdf_file = Path("docs/Executive_Presentation.pdf")
    
    print("🔄 Converting Executive Presentation to PDF...")
    
    # Try Pandoc first (best quality)
    try:
        subprocess.run([
            "pandoc",
            str(md_file),
            "-o", str(pdf_file),
            "--pdf-engine=xelatex",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "colorlinks=true",
            "--toc",
            "--toc-depth=2"
        ], check=True)
        print(f"✅ PDF created: {pdf_file}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Pandoc not found or failed.")
    
    # Try markdown-pdf
    try:
        subprocess.run([
            "markdown-pdf",
            str(md_file),
            "-o", str(pdf_file)
        ], check=True)
        print(f"✅ PDF created: {pdf_file}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  markdown-pdf not found or failed.")
    
    print("\n📋 Manual conversion required:")
    print("   See instructions in docs/convert_to_pdf.py")
    print(f"   Source: {md_file}")
    return False

if __name__ == "__main__":
    convert_to_pdf()
