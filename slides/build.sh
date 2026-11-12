#!/usr/bin/env bash
# Render every slide deck under slides/ and assemble it for publishing.
#
#   slides/build.sh [OUT]     # OUT defaults to public/slides
#
# Each deck (a directory holding index.qmd) is rendered with Quarto alone (no Python) and copied to OUT/<deck>/
# together with the files it loads at runtime: index_files/, assets/ and style.css. CI runs this after `mkdocs build`,
# so the decks are published under the docs site (https://aidotse.github.io/cafl4ds/slides/<deck>/).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$(realpath -m "${1:-public/slides}")"

shopt -s nullglob
decks=("$here"/*/index.qmd)
if [ ${#decks[@]} -eq 0 ]; then
  echo "no decks found under $here" >&2
  exit 1
fi

for qmd in "${decks[@]}"; do
  deck="$(dirname "$qmd")"
  name="$(basename "$deck")"
  echo "rendering $name"
  (cd "$deck" && quarto render index.qmd)
  rm -rf "${out:?}/${name:?}"
  mkdir -p "$out/$name"
  cp -r "$deck/index.html" "$deck/index_files" "$deck/assets" "$deck/style.css" "$out/$name/"
done
echo "decks written to $out"
