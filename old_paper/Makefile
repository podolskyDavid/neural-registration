# Makefile for compiling the thesis

# Configuration
MAIN = main
OUTPUT_DIR = build
LATEX = pdflatex
BIBER = biber
LATEX_OPTS = -interaction=nonstopmode -output-directory=$(OUTPUT_DIR)

# Main targets
all: $(OUTPUT_DIR)/$(MAIN).pdf

$(OUTPUT_DIR)/$(MAIN).pdf: $(MAIN).tex $(wildcard chapters/*.tex) references.bib
	mkdir -p $(OUTPUT_DIR)
	$(LATEX) $(LATEX_OPTS) $(MAIN)
	$(BIBER) $(OUTPUT_DIR)/$(MAIN)
	$(LATEX) $(LATEX_OPTS) $(MAIN)
	$(LATEX) $(LATEX_OPTS) $(MAIN)

# Clean targets
clean:
	rm -rf $(OUTPUT_DIR)

# View the PDF
view: $(OUTPUT_DIR)/$(MAIN).pdf
	open $(OUTPUT_DIR)/$(MAIN).pdf

# Continuous compilation
watch:
	while true; do \
		make all; \
		sleep 2; \
		inotifywait -e modify -r . --include ".*\.(tex|bib)"; \
	done

.PHONY: all clean view watch 