TEXSRC = $(wildcard ./*.tex)

all: report clean

report: report.pdf

report.pdf: biblio.bib $(TEXSRC)
	pdflatex report
	bibtex report
	pdflatex report
	pdflatex report

clean: 
	\rm -f *.dvi *.aux *.ps *~ *.log *.blg *.bbl *.fls *.fdb_latexmk *.synctex.gz *.out
