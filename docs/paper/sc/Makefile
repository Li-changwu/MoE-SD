TARGET=main

all:
	pdflatex -interaction=nonstopmode -halt-on-error $(TARGET).tex
	bibtex $(TARGET)
	pdflatex -interaction=nonstopmode -halt-on-error $(TARGET).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(TARGET).tex

clean:
	rm -f $(TARGET).aux $(TARGET).bbl $(TARGET).blg $(TARGET).log $(TARGET).out $(TARGET).pdf