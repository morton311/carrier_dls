#!/usr/bin/env bash
# Build a LaTeX document the way the LaTeX Workshop remote settings do, then auto-clean on success.
#
#   scripts/latex_build.sh papers/scitech_2027/main.tex
#
# Mirrors ~/.vscode-server/data/Machine/settings.json: latexmk -synctex=1 -interaction=nonstopmode -file-line-error
# -pdf -outdir=./latex_output, run from the document's folder; autoClean.run = onSucceeded with clean.subfolder
# enabled, using LaTeX Workshop's default clean file types. The PDF and main.synctex.gz stay in ./latex_output.
set -euo pipefail
doc=$(realpath "$1")
cd "$(dirname "$doc")"
name=$(basename "$doc" .tex)
if latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf -outdir=./latex_output "$name.tex"; then
    for d in . ./latex_output; do                      # document folder: top level only, so data in figs/ is safe
        find "$d" $( [ "$d" = . ] && echo -maxdepth 1 ) -type f \( -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.idx' -o -name '*.ind' \
            -o -name '*.lof' -o -name '*.lot' -o -name '*.out' -o -name '*.toc' -o -name '*.acn' -o -name '*.acr' \
            -o -name '*.alg' -o -name '*.glg' -o -name '*.glo' -o -name '*.gls' -o -name '*.fls' -o -name '*.log' \
            -o -name '*.fdb_latexmk' -o -name '*.snm' -o -name '*.nav' -o -name '*.vrb' -o -name '*.bcf' \
            -o -name '*.run.xml' -o -name '*.synctex(busy)' -o -name '*.synctex.gz(busy)' \) \
            -delete
    done
    find ./latex_output -mindepth 1 -type d -empty -delete
    echo "built $(pwd)/latex_output/$name.pdf"
else
    echo "build failed; auxiliary files kept in $(pwd)/latex_output" >&2
    exit 1
fi
