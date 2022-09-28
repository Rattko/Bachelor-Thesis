all: poster.pdf

poster.pdf: poster.tex
	pdflatex $<
	pdflatex $<

clean:
	rm -rf poster.pdf poster.aux poster.log
