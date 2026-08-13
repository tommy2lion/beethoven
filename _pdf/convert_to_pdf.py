"""Convert beethoven markdown docs to PDF via Edge headless print-to-pdf.

LaTeX math: rendered to SVG via matplotlib (offline, no CDN needed).
"""

import base64, html, io, pathlib, re, subprocess, sys, markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PROJECT_DIR = pathlib.Path(__file__).parent.parent
DOCS_DIR = PROJECT_DIR / "docs"
OUT_DIR = PROJECT_DIR / "_pdf"
OUT_DIR.mkdir(exist_ok=True)

CSS = """
@page { size: A4; margin: 16mm 14mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei","Segoe UI",-apple-system,Arial,sans-serif;
       font-size: 12px; line-height: 1.7; color: #1f2937; }
h1 { font-size: 22px; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; margin: 18px 0 12px; }
h2 { font-size: 18px; border-bottom: 1px solid #eee; padding-bottom: 4px; margin: 20px 0 10px; }
h3 { font-size: 15px; margin: 16px 0 8px; color: #111827; }
h4 { font-size: 13px; margin: 12px 0 6px; }
p, li { margin: 4px 0; }
code { font-family: ui-monospace,Consolas,Menlo,monospace; background: #f3f4f6;
       padding: 1px 4px; border-radius: 3px; font-size: 11px; }
pre { background: #f6f8fa; border: 1px solid #e5e7eb; border-radius: 6px;
      padding: 10px; overflow-x: auto; font-size: 10.5px; white-space: pre-wrap; word-break: break-word; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 11px; }
th, td { border: 1px solid #d1d5db; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f3f4f6; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
blockquote { border-left: 3px solid #cbd5e1; margin: 8px 0; padding: 2px 12px;
             color: #475569; background: #f8fafc; }
img.latex-svg { vertical-align: middle; display: inline; margin: 2px 0; }
div.latex-svg { text-align: center; margin: 8px 0; }
img { max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px; margin: 6px 0; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 16px 0; }
a { color: #2563eb; text-decoration: none; }
"""


def latex_to_svg(latex_code, display=False):
    """Render LaTeX to SVG using matplotlib, return base64 data URI."""
    try:
        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        ax.axis('off')

        # Use matplotlib's LaTeX rendering
        text = ax.text(0, 0, f"${latex_code}$" if not display else f"$${latex_code}$$",
                       fontsize=11, ha='left', va='bottom',
                       usetex=False)  # usetex=False means mathtext (built-in)

        # Render to figure
        fig.canvas.draw()

        # Get the text's bounding box
        bbox = text.get_window_extent(fig.canvas.get_renderer())
        bbox = bbox.expanded(1.1, 1.2)  # add padding

        # Set figure size to bbox
        fig.set_size_inches(bbox.width / fig.dpi, bbox.height / fig.dpi)

        # Save to SVG
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight',
                    pad_inches=0.05, transparent=True)
        plt.close(fig)

        svg_data = buf.getvalue().decode('utf-8')
        b64 = base64.b64encode(svg_data.encode('utf-8')).decode()
        css_class = "latex-svg"
        tag = "div" if display else "span"
        return f'<{tag} class="{css_class}"><img class="{css_class}" src="data:image/svg+xml;base64,{b64}" alt="{html.escape(latex_code)}"/></{tag}>'
    except Exception as e:
        # Fallback: just show the LaTeX source in italics
        return f'<i>[{html.escape(latex_code)}]</i>'


def render_latex(text):
    """
    Replace LaTeX math expressions with SVG images.
    Handles both $$...$$ (display) and $...$ (inline).
    """
    # Display math: $$...$$
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: latex_to_svg(m.group(1), display=True), text, flags=re.S)
    # Inline math: $...$ (but not $$ which we already handled)
    text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', lambda m: latex_to_svg(m.group(1), display=False), text)
    return text


def convert(md_path: pathlib.Path) -> pathlib.Path:
    text = md_path.read_text(encoding="utf-8")

    # stash mermaid blocks
    blocks: list[str] = []
    def _stash(m):
        blocks.append(m.group(1))
        return f"\n\nMERMAIDSTASH{len(blocks) - 1}ENDSTASH\n\n"
    text = re.sub(r"```mermaid\n(.*?)```", _stash, text, flags=re.S)

    # Convert markdown to HTML
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "toc"]
    )

    # Render LaTeX (before mermaid restore, after markdown)
    body = render_latex(body)

    # Restore mermaid
    for i, code in enumerate(blocks):
        body = body.replace(
            f"<p>MERMAIDSTASH{i}ENDSTASH</p>",
            f'<pre class="mermaid">{html.escape(code)}</pre>',
        )

    # inline-embed local images
    def _embed(m):
        src = m.group(1)
        p = (md_path.parent / src).resolve()
        if p.exists() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".svg"):
            b64 = base64.b64encode(p.read_bytes()).decode()
            ext = "svg+xml" if p.suffix.lower() == ".svg" else p.suffix.lstrip(".")
            return f'src="data:image/{ext};base64,{b64}"'
        return m.group(0)
    body = re.sub(r'src="([^"]+)"', _embed, body)

    full = (
        f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )

    # HTML intermediate
    html_path = OUT_DIR / f"{md_path.stem}.html"
    html_path.write_text(full, encoding="utf-8")

    # PDF via Edge headless
    pdf_path = OUT_DIR / f"{md_path.stem}.pdf"
    cmd = [
        EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        "--virtual-time-budget=15000",
        html_path.as_uri(),
    ]
    subprocess.run(cmd, check=True, timeout=120)
    print(f"  OK  {pdf_path.name}  ({pdf_path.stat().st_size // 1024} KB)")
    return pdf_path


if __name__ == "__main__":
    files = [
        DOCS_DIR / "technical_foundation.md",
        DOCS_DIR / "approach_1.md",
        DOCS_DIR / "approach_2.md",
        DOCS_DIR / "roadmap.md",
        PROJECT_DIR / "README.md",
        DOCS_DIR / "development_summary.md",
        DOCS_DIR / "math_foundation.md",
    ]

    print(f"Converting {len(files)} documents to PDF...\n")
    for f in files:
        if f.exists():
            print(f"  Input: {f.name}")
            convert(f)
        else:
            print(f"  SKIP: {f.name} (not found)")

    print(f"\nDone! {len(list(OUT_DIR.glob('*.pdf')))} PDFs in {OUT_DIR}")
