SRC=src/fast_align.cc src/ttables.h src/da.h src/corpus.h src/gmm.h src/kmeans.h

fast_align: $(SRC)
	g++ -Werror -Wall -O3 -I. src/fast_align.cc -o $@

debug: $(SRC)
	g++ -Werror -Wall -g -I. src/fast_align.cc -o fast_align.debug
