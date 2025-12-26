"""Simple Markdown to HTML converter with print-friendly CSS"""

import markdown
from pathlib import Path

def convert_markdown_to_html():
    """Convert the executive presentation to a styled HTML that can be printed to PDF."""
    
    # Paths
    md_file = Path("docs/Executive_Presentation.md")
    html_file = Path("docs/Executive_Presentation.html")
    
    print(f"📄 Reading {md_file}...")
    
    # Read markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'nl2br']
    )
    
    # Create full HTML with professional CSS
    full_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion-Based Discovery for Audible - Executive Presentation</title>
    <style>
        @media print {
            @page {
                size: letter;
                margin: 0.75in;
            }
            
            body {
                margin: 0;
                padding: 0;
            }
            
            h1, h2, h3 {
                page-break-after: avoid;
            }
            
            table, figure {
                page-break-inside: avoid;
            }
        }
        
        @media screen {
            body {
                max-width: 8.5in;
                margin: 0 auto;
                padding: 1in;
                background: #f5f5f5;
            }
            
            .container {
                background: white;
                padding: 1in;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }
        
        h1 {
            color: #1a365d;
            font-size: 28pt;
            font-weight: 700;
            margin: 0 0 30pt 0;
            padding-bottom: 15pt;
            border-bottom: 4px solid #2563eb;
        }
        
        h2 {
            color: #1e40af;
            font-size: 20pt;
            font-weight: 600;
            margin: 36pt 0 18pt 0;
            padding-bottom: 8pt;
            border-bottom: 2px solid #e5e7eb;
        }
        
        h3 {
            color: #1e3a8a;
            font-size: 16pt;
            font-weight: 600;
            margin: 24pt 0 12pt 0;
        }
        
        h4 {
            color: #1e40af;
            font-size: 13pt;
            font-weight: 600;
            margin: 18pt 0 10pt 0;
        }
        
        p {
            margin: 0 0 12pt 0;
            text-align: justify;
        }
        
        strong {
            color: #1e3a8a;
            font-weight: 600;
        }
        
        em {
            color: #4b5563;
            font-style: italic;
        }
        
        ul, ol {
            margin: 12pt 0;
            padding-left: 30pt;
        }
        
        li {
            margin-bottom: 8pt;
        }
        
        ul ul, ol ol {
            margin-top: 6pt;
            margin-bottom: 6pt;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20pt 0;
            font-size: 10pt;
        }
        
        thead {
            background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
        }
        
        th {
            background-color: #2563eb;
            color: white;
            padding: 12pt 10pt;
            text-align: left;
            font-weight: 600;
            border: 1px solid #1e40af;
        }
        
        td {
            border: 1px solid #d1d5db;
            padding: 10pt;
            vertical-align: top;
        }
        
        tbody tr:nth-child(even) {
            background-color: #f9fafb;
        }
        
        tbody tr:hover {
            background-color: #eff6ff;
        }
        
        code {
            background-color: #f3f4f6;
            color: #dc2626;
            padding: 2pt 6pt;
            border-radius: 4pt;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 10pt;
        }
        
        pre {
            background-color: #1f2937;
            color: #f3f4f6;
            padding: 15pt;
            border-radius: 6pt;
            overflow-x: auto;
            margin: 15pt 0;
        }
        
        pre code {
            background: none;
            color: inherit;
            padding: 0;
        }
        
        blockquote {
            border-left: 4px solid #2563eb;
            background-color: #eff6ff;
            padding: 12pt 15pt;
            margin: 15pt 0;
            color: #1e40af;
            font-style: italic;
        }
        
        hr {
            border: none;
            border-top: 2px solid #e5e7eb;
            margin: 30pt 0;
        }
        
        a {
            color: #2563eb;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .page-break {
            page-break-after: always;
        }
        
        /* Header styling */
        .header {
            text-align: center;
            margin-bottom: 40pt;
        }
        
        /* Footer for print */
        @media print {
            .no-print {
                display: none;
            }
        }
        
        .print-button {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14pt;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        
        .print-button:hover {
            background: #1e40af;
        }
    </style>
</head>
<body>
    <button class="print-button no-print" onclick="window.print()">🖨️ Save as PDF</button>
    <div class="container">
""" + html_body + """
    </div>
    
    <script>
        // Instructions for saving as PDF
        console.log('To save as PDF: Click the Print button above or press Ctrl+P, then choose "Save as PDF"');
    </script>
</body>
</html>"""
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"✅ HTML created: {html_file}")
    print(f"📊 File size: {html_file.stat().st_size / 1024:.1f} KB")
    print(f"\n📝 To create PDF:")
    print(f"   1. Open {html_file} in your browser")
    print(f"   2. Click the 'Save as PDF' button (or press Ctrl+P)")
    print(f"   3. Select 'Microsoft Print to PDF' or 'Save as PDF'")
    print(f"   4. Save as 'Executive_Presentation.pdf'")
    
    return html_file

if __name__ == "__main__":
    try:
        html_path = convert_markdown_to_html()
        print(f"\n🎉 Success! Open: {html_path.absolute()}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
