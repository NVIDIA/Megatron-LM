default: cases.png scaling-mp.png scaling-dp.png

# for some reason the size option to convert in scaling.tex doesn't work, manually do it after
cases.png scaling-mp.png scaling-dp.png: tables.tex
	latex --shell-escape $<
	convert tables-1.png -resize 650 cases.png
	convert tables-2.png -resize 600 scaling-mp.png
	convert tables-3.png -resize 350 scaling-dp.png

clean:
	rm -rf *.aux *.log *.dvi *.ps
	rm -rf tables-*.png
