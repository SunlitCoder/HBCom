# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math
import numpy as np

def _get_ngrams(segment, max_n):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_n: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_n in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_n + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def corpus_bleu(candidate_corpus,reference_corpus, max_n=4,weights=[0.25] * 4,
                 smooth=True):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      candidate_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_n: the maximum n-gram we want to use. E.g. if max_n=3, we will use unigrams,
            bigrams and trigrams
        weights: a list of weights used for each n-gram category (uniform by default)
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    assert max_n == len(weights), 'Length of the "weights" list has be equal to max_n'
    assert len(candidate_corpus) == len(reference_corpus), \
        'The length of candidate and reference corpus should be the same'
    matches_by_order = [0] * max_n
    possible_matches_by_order = [0] * max_n
    reference_length = 0
    candidate_length = 0
    for (references, translation) in zip(reference_corpus,
                                         candidate_corpus):
        reference_length += min(len(r) for r in references)
        candidate_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_n)
        candidate_ngram_counts = _get_ngrams(translation, max_n)
        overlap = candidate_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_n + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_n
    for i in range(0, max_n):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        # p_log_sum = sum((1. / max_n) * math.log(p) for p in precisions)
        p_log_sum =np.sum(np.array(weights)*np.log(np.array(precisions)))
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(candidate_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio) if ratio!=0 else math.exp(1-math.inf)

    bleu = geo_mean * bp
    # return (bleu, precisions, bp, ratio, translation_length, reference_length)
    return bleu


if __name__=='__main__':
    def get_google_sent_bleu(preds, refs):
        '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu'''
        if not isinstance(refs[0][0], list):  # 必不可少
            refs = [[seq] for seq in refs]
        scores = [corpus_bleu([pred], [ref], max_n=4, weights=[0.25, 0.25, 0.25, 0.25], smooth=True)
                  for pred, ref in zip(preds, refs)]
        return sum(scores) / len(scores) * 100

    candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence'],['yes']]
    references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']],[['yes']]]
    print(corpus_bleu(candidate_corpus,references_corpus,smooth=True))

    candidate_corpus = [[' ']]
    references_corpus = [[['yes']]]
    print(corpus_bleu(candidate_corpus, references_corpus, smooth=True))
    print(get_google_sent_bleu(candidate_corpus,references_corpus))