# Bachelor's Thesis: NeRF-based Intraoperative Registration

This directory contains the LaTeX source files for the bachelor's thesis on NeRF-based Intraoperative Registration.

## Directory Structure

- `main.tex`: Main LaTeX file that includes all other files
- `titlepage.tex`: Title page of the thesis
- `references.bib`: Bibliography file in BibTeX format
- `chapters/`: Directory containing individual chapter files
  - `introduction.tex`: Introduction chapter
  - `background.tex`: Background and related work
  - `methodology.tex`: Methodology chapter
  - `implementation.tex`: Implementation details
  - `experiments.tex`: Experimental setup
  - `results.tex`: Results and analysis
  - `discussion.tex`: Discussion of results
  - `conclusion.tex`: Conclusion chapter
  - `appendix.tex`: Appendices
- `figures/`: Directory for figures and images
- `Makefile`: Makefile for compiling the thesis

## Compilation

To compile the thesis, you need a LaTeX distribution installed (e.g., TeX Live, MiKTeX).

### Using the Makefile

```bash
# Compile the thesis
make

# View the compiled PDF
make view

# Clean build files
make clean

# Watch for changes and recompile automatically (requires inotifywait)
make watch
```

### Manual Compilation

If you don't want to use the Makefile, you can compile manually:

```bash
mkdir -p build
pdflatex -interaction=nonstopmode -output-directory=build main
biber build/main
pdflatex -interaction=nonstopmode -output-directory=build main
pdflatex -interaction=nonstopmode -output-directory=build main
```

## Required LaTeX Packages

The thesis requires the following LaTeX packages:

- inputenc
- fontenc
- lmodern
- amsmath, amssymb, amsfonts
- graphicx
- hyperref
- booktabs
- caption, subcaption
- algorithm, algpseudocode
- listings
- xcolor
- biblatex
- csquotes

Most of these packages are included in standard LaTeX distributions.

## Customization

To customize the thesis:

1. Edit `titlepage.tex` to update your name, department, university, and supervisor information
2. Modify the abstract in `main.tex`
3. Update the content in the chapter files
4. Add your references to `references.bib`
5. Add your figures to the `figures/` directory 