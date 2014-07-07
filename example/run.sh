#!/bin/sh

echo 'Forward alignment'
../fast_align -i text.en-fr -d -o -v >forward.align
echo "==> diff with example file: forward.align.example ...\n"
diff forward.align forward.align.example

echo 'Reverse alignment'
../fast_align -i text.en-fr -d -o -v -r >reverse.align
echo "==> diff with example file: reverse.align.example ...\n"
diff reverse.align reverse.align.example

#./atools -i forward.align -j reverse.align -c grow-diag-final-and > align.gdfa