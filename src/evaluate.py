# Generated: 2025-08-13
# File: src/evaluate.py

from typing import List
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def bleu_score_corpus(refs: List[List[List[str]]], hyps: List[List[str]]) -> float:
    """
    refs: list over samples -> list over references -> list of tokens
    hyps: list over samples -> list of tokens
    """
    ch = SmoothingFunction()
    return corpus_bleu(refs, hyps, smoothing_function=ch.method1)

if __name__ == "__main__":
    refs = [[["a","dog","on","grass"]], [["a","cat"]]]
    hyps = [["a","dog","on","grass"], ["a","cat"]]
    print("BLEU:", bleu_score_corpus(refs, hyps))
