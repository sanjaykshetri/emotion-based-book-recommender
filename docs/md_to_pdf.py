"""Convert Executive Presentation Markdown to PDF with WeasyPrint"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path

def convert_markdown_to_pdf():
    """Convert the executive presentation to a styled PDF."""
    
    # Paths
    md_file = Path("docs/Executive_Presentation.md")
    pdf_file = Path("docs/Executive_Presentation.pdf")
    
    print(f"📄 Reading {md_file}...")
    
    # Read markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite']
    )
    
    # Add CSS styling for professional look
    css_style = CSS(string="""
        @page {
            size: Letter;
            margin: 1in;
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 9pt;
                color: #666;
            }
        }
        
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 24pt;
            margin-top: 0;
            margin-bottom: 20pt;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10pt;
        }
        
        h2 {
            color: #2c3e50;
            font-size: 18pt;
            margin-top: 24pt;
            margin-bottom: 12pt;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 6pt;
        }
        
        h3 {
            color: #34495e;
            font-size: 14pt;
            margin-top: 18pt;
            margin-bottom: 10pt;
        }
        
        h4 {
            color: #34495e;
            font-size: 12pt;
            margin-top: 14pt;
            margin-bottom: 8pt;
        }
        
        p {
            margin-bottom: 10pt;
            text-align: justify;
        }
        
        strong {
            color: #2c3e50;
            font-weight: 600;
        }
        
        em {
            color: #555;
        }
        
        ul, ol {
            margin-bottom: 12pt;
            padding-left: 25pt;
        }
        
        li {
            margin-bottom: 6pt;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15pt 0;
            font-size: 10pt;
        }
        
        th {
            background-color: #3498db;
            color: white;
            padding: 8pt;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            border: 1px solid #ddd;
            padding: 8pt;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        code {
            background-color: #f4f4f4;
            padding: 2pt 4pt;
            border-radius: 3pt;
            font-family: 'Consolas', monospace;
            font-size: 10pt;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 15pt;
            margin-left: 0;
            color: #555;
            font-style: italic;
        }
        
        hr {
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 20pt 0;
        }
    """)
    
    # Wrap HTML with proper structure
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Emotion-Based Discovery for Audible - Executive Presentation</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    print("🎨 Converting to PDF with styling...")
    
    # Convert to PDF
    HTML(string=full_html).write_pdf(
        pdf_file,
        stylesheets=[css_style]
    )
    
    print(f"✅ PDF created: {pdf_file}")
    print(f"📊 File size: {pdf_file.stat().st_size / 1024:.1f} KB")
    return pdf_file

if __name__ == "__main__":
    try:
        pdf_path = convert_markdown_to_pdf()
        print(f"\n🎉 Success! Open: {pdf_path.absolute()}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
