# Data statistics with XLM-R in English

## Mean statistics of train set

| label      |   flesch_reading_ease |   flesch_kincaid_grade |   gunning_fog |   automated_readability_index |   coleman_liau_index |   linsear_write_formula |   dale_chall_readability_score |   text_standard |   spache_readability |   mcalpine_eflaw |   reading_time |   syllable_count |   lexicon_count |   sentence_count |   char_count |   letter_count |   polysyllabcount |   monosyllabcount |   difficult_words |   get_avg_syllables |
|:-----------|----------------------:|-----------------------:|--------------:|------------------------------:|---------------------:|------------------------:|-------------------------------:|----------------:|---------------------:|-----------------:|---------------:|-----------------:|----------------:|-----------------:|-------------:|---------------:|------------------:|------------------:|------------------:|--------------------:|
| approx. 1b |               76.4168 |                7.04656 |       9.48853 |                       8.94898 |              7.6839  |                 9.97955 |                        8.52617 |         8.25605 |              4.62457 |          25.6494 |        4.19975 |          83.7351 |         62.1047 |          3.80786 |      285.893 |        277.835 |           5.11652 |           47.2267 |           9.63953 |             1.47011 |
| approx. 7b |               77.9787 |                6.64857 |       9.10176 |                       8.38174 |              7.33212 |                 9.46922 |                        8.39515 |         7.92862 |              4.48113 |          24.8574 |        4.07917 |          81.7046 |         61.1124 |          3.92649 |      277.684 |        269.829 |           4.89812 |           46.9196 |           9.08723 |             1.45718 |



## Mean statistics of test set on label

| label      |   flesch_reading_ease |   flesch_kincaid_grade |   gunning_fog |   automated_readability_index |   coleman_liau_index |   linsear_write_formula |   dale_chall_readability_score |   text_standard |   spache_readability |   mcalpine_eflaw |   reading_time |   syllable_count |   lexicon_count |   sentence_count |   char_count |   letter_count |   polysyllabcount |   monosyllabcount |   difficult_words |   get_avg_syllables |
|:-----------|----------------------:|-----------------------:|--------------:|------------------------------:|---------------------:|------------------------:|-------------------------------:|----------------:|---------------------:|-----------------:|---------------:|-----------------:|----------------:|-----------------:|-------------:|---------------:|------------------:|------------------:|------------------:|--------------------:|
| approx. 1b |               76.0637 |                7.23313 |        9.7    |                       9.19553 |              7.66338 |                10.321   |                        8.61327 |         8.45224 |              4.71812 |          26.3958 |        4.14062 |          82.5003 |         61.259  |          3.67753 |      281.865 |        274.029 |           4.98067 |           46.5527 |           9.62562 |             1.46759 |
| approx. 7b |               76.7766 |                6.98415 |        9.4332 |                       8.82693 |              7.5413  |                 9.93584 |                        8.43772 |         8.09632 |              4.59566 |          25.8136 |        4.16193 |          83.231  |         62.0476 |          3.92077 |      283.317 |        275.339 |           5.0523  |           47.535  |           9.35267 |             1.46683 |



## Mean statistics of test set on prediction

| hyp_label   |   flesch_reading_ease |   flesch_kincaid_grade |   gunning_fog |   automated_readability_index |   coleman_liau_index |   linsear_write_formula |   dale_chall_readability_score |   text_standard |   spache_readability |   mcalpine_eflaw |   reading_time |   syllable_count |   lexicon_count |   sentence_count |   char_count |   letter_count |   polysyllabcount |   monosyllabcount |   difficult_words |   get_avg_syllables |
|:------------|----------------------:|-----------------------:|--------------:|------------------------------:|---------------------:|------------------------:|-------------------------------:|----------------:|---------------------:|-----------------:|---------------:|-----------------:|----------------:|-----------------:|-------------:|---------------:|------------------:|------------------:|------------------:|--------------------:|
| approx. 1b  |               76.4316 |                7.10465 |       9.56232 |                       9.00532 |              7.60038 |                 10.1223 |                        8.52268 |         8.26857 |              4.65493 |          26.0954 |        4.15162 |          82.8773 |          61.666 |          3.80305 |      282.615 |        274.705 |           5.01764 |           47.0596 |           9.48477 |              1.4672 |



## Mean statistics of test set on label x prediction

| label X hyp_label                  |   flesch_reading_ease |   flesch_kincaid_grade |   gunning_fog |   automated_readability_index |   coleman_liau_index |   linsear_write_formula |   dale_chall_readability_score |   text_standard |   spache_readability |   mcalpine_eflaw |   reading_time |   syllable_count |   lexicon_count |   sentence_count |   char_count |   letter_count |   polysyllabcount |   monosyllabcount |   difficult_words |   get_avg_syllables |
|:-----------------------------------|----------------------:|-----------------------:|--------------:|------------------------------:|---------------------:|------------------------:|-------------------------------:|----------------:|---------------------:|-----------------:|---------------:|-----------------:|----------------:|-----------------:|-------------:|---------------:|------------------:|------------------:|------------------:|--------------------:|
| approx. 1b--CORRECT                |               76.0637 |                7.23313 |        9.7    |                       9.19553 |              7.66338 |                10.321   |                        8.61327 |         8.45224 |              4.71812 |          26.3958 |        4.14062 |          82.5003 |         61.259  |          3.67753 |      281.865 |        274.029 |           4.98067 |           46.5527 |           9.62562 |             1.46759 |
| approx. 7b-->approx. 1b--INCORRECT |               76.7766 |                6.98415 |        9.4332 |                       8.82693 |              7.5413  |                 9.93584 |                        8.43772 |         8.09632 |              4.59566 |          25.8136 |        4.16193 |          83.231  |         62.0476 |          3.92077 |      283.317 |        275.339 |           5.0523  |           47.535  |           9.35267 |             1.46683 |

