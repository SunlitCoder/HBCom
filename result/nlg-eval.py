from nlgeval import compute_metrics

metrics_dict = compute_metrics(hypothesis="clean_result/clean_result_hyp.csv",
                               references=["clean_result/clean_result_ref.csv"], no_skipthoughts=True,
                               no_glove=True)