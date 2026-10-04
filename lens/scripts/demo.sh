#!/usr/bin/env bash
# A tour of lens against the bundled corpus.
set -euo pipefail
cd "$(dirname "$0")/.."

run() { echo; echo "--- $1"; shift; python3 -m lens.cli "$@"; }

run "a rare literal token -- the case BM25 is for" \
    search "ANN-4417" -k 3

run "a paraphrase sharing no words -- the case embeddings are for" \
    search "why does splitting text into pieces that overlap help" -k 3

run "the same query through each retriever, for comparison" \
    search "deleting vectors from an index" --retriever lexical -k 3
python3 -m lens.cli search "deleting vectors from an index" --retriever dense -k 3

run "a cited answer, with the exact prompt the model received" \
    ask "what does efSearch control" -k 3 --show-prompt

run "evaluating one configuration, including the queries that failed" \
    eval --retriever hybrid

run "comparing fifteen configurations" \
    sweep

echo
echo "done."
