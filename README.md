### Paper "Bash Command Comment Generation via Multi-Scale Heterogeneous Feature Fusion"

1. **Model Structure**
![HBCom Structure](https://github.com/user-attachments/assets/4225c898-8325-4aa2-988c-436616865366)

3. **Directory Introduction**

   - **data**: stores raw and intermediate data files.
   - **lib**: contains models and modules used in the project.
   - **result**: stores the result data.
   - **src**: contains the files related to the training process and steps.
   - **util**: contains utility packages used for data processing and training procedures.

4. **Model Training Steps**

   1. Navigate to the `src` folder.
   2. Run `python s1_preprocessor.py` to preprocess the data.
   3. Execute `python s2_model.py` to train the model.

5. **Result Evaluation**

   1. Go to the `result` folder.
   2. Run `python nlg-eval.py` to evaluate the generated results using NLG-Eval metrics.
   3. Execute `python bleu.py` to evaluate the generated results using BLEU.
