pandoc \
    --pdf-engine=xelatex \
    -H head.tex \
    --highlight-style kate \
    -V geometry:a4paper \
    -V geometry:margin=4cm \
    -V linkcolor:blue \
    ex4.md -o ex4.pdf

