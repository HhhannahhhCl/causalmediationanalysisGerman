# causalmediationanalysisGerman
This repository provides the necessary parts to carry out causal mediation analysis in German GPT-2 models.

## Data
The data is based on the WinoBias data by Zhao et al. (2018) meant to test NLP models for gender bias within the task of pronoun resolution. It was created based on the German labour market and formulated by a native speaker. Needed for the analysis are only the stereotypical sentences. These are available in masculine generics, feminine generics, and gender-neutral language using an asterisk.

## Code
The code repository is taken from Vig et al. (2020) and adjusted to newer versions of the transformers library as well as pytorch. Furthermore, adjustments were made for the code to be suitable for attention interventions on German GPT-2 models.

## Results
The results are saved in json files and provide the unfiltered and filtered results for German GPT-2 (Schweter 2021a), German GPT-2 larger (Schweter 2021b), GerPT-2, and GerPT-2 large (Minixhofer 2020). 

## Bibliography
Minixhofer, B. (2020, 12). GerPT2: German large and small versions of GPT2. Retrieved from https://github.com/bminixhofer/gerpt2 ([Online; accessed 09-January-2024]) 
Schweter, S. (2021a, August). German gpt-2 model. Retrieved from https://github.com/stefan-it/german-gpt2 ([Online; accessed 09-January-2024]) 
Schweter, S. (2021b, September). German gpt-2 model. Retrieved from https://huggingface.co/stefan-it/german-gpt2-larger ([Online; accessed 09-January-2024])
Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Sakenis, S., . . . Shieber, S. (2020). Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias.
Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K.-W. (2018, June). Gender bias in coreference resolution: Evaluation and debiasing methods. In Proceedings of the 2018 conference of the north American chapter of the association for computational linguistics: Human language technologies, volume 2 (short papers) (pp. 15–20). New Orleans, Louisiana: Association for Computational Linguistics. 
