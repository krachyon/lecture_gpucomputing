pandoc \
    --pdf-engine=xelatex \
    -H head.tex \
    --highlight-style kate \
    -V geometry:a4paper \
    -V geometry:margin=4cm \
    -V linkcolor:blue \
    exercise3.md -o ex3.pdf

