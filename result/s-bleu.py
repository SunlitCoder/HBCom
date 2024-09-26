import pandas as pd

from eval.translate_metric import get_nltk33_sent_bleu as get_sent_bleu

hyp_path= 'clean_result/clean_result_hyp.csv'
ref_path= 'clean_result/clean_result_ref.csv'

hyp_df = pd.read_csv(hyp_path)
ref_df = pd.read_csv(ref_path)

hyp_list = []
ref_list = []

for text in hyp_df.values:
    hyp_list.append(text[0].split())

for text in ref_df.values:
    ref_list.append(text[0].split())

print(get_sent_bleu.__name__, ':', get_sent_bleu(hyp_list, ref_list))