all: plot.pdf

plot.pdf: data plot.gpi
	cat plot.gpi | sed 's/OUT/$@/; s/INPUT/$</' | tee log | pyxplot

plot.gpi: Makefile
	echo 'set term pdf' > $@
	echo 'set title "Plot"' >> $@
	echo 'set output "OUT"' >> $@
	echo 'set xlabel "x"'       >> $@
	echo 'set ylabel "y"' >> $@
	echo 'set key outside' >> $@
	echo 'set logscale x' >> $@
	echo 'plot \' >> $@
	echo ' "INPUT" index 0 using 1:2 notitle with points color black ' >> $@

clean:
	$(RM) -r *.pdf

.PHONY: clean
