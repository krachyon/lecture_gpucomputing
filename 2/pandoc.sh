pandoc \
    --pdf-engine=xelatex \
    -H head.tex \
    --highlight-style kate \
    -V geometry:a4paper \
    -V geometry:margin=4cm \
    -V linkcolor:blue \
    exercise2.md -o ex2.pdf

