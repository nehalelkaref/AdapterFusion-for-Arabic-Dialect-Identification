# AdapterFusion for Arabic Dialect Identification 🤖
<img width="1259" alt="Screenshot 2023-12-18 at 15 50 10" src="https://github.com/nehalelkaref/Arabic-Dialect-Identification/assets/16616024/5d1c9b9b-59bc-4efe-a66b-83fbd2650842">

[AdapterFusion](https://arxiv.org/abs/2005.00247) is a learning algorithm that combines task-relevant information contained within [adapters](https://arxiv.org/abs/1902.00751) to fine-tune a target task using multiple resources.
We employ AdapterFusion on a classsic Arabic NLP problem, *Dialect Identification*, and deploy the resulting model on HuggingFace Spaces🤗

<sub>_**This project is a fraction of a Msc degree dedicated to Arabic Dialect Identification that was supervised by [Prof. Dr. Mervat Abouelkhair](https://github.com/mervatkheir)**_</sub>

### Requriements for Fusion:
- An AdapterFusion model requires *N* adapters
-  Each *n* adapter is fine-tuned on task-relevant data
- In our case, we fine-tune, we fine-tune a total of *8* adapters on the data below covering MSA/DA, Region, Country and Province Levels
- For each diatopic variation an AdapterFusion (AF) model is created, hence AF for each Region, Country, Province and MSA/DA
- We share one of the best performing AF models on Regional Dialect Identification here: [HuggingFace Spaces🤗](https://huggingface.co/spaces/nehalelkaref/RegionClassifier)


### Data and Domains:
- [Arabic Online Commentary (AOC)](https://aclanthology.org/P11-2007/), _Newspaper Commentary_
- [NADI 2020](https://aclanthology.org/2020.wanlp-1.9/), _Tweets_
- [MADAR](https://aclanthology.org/L18-1535.pdf), _Tweets_
- [ArSarcasm](https://paperswithcode.com/paper/from-arabic-sentiment-analysis-to-sarcasm), _Tweets_
- [QADI](https://aclanthology.org/2021.wanlp-1.1/), _Tweets_

<sub>_**Code for adapter-training and adapterfusion available in repo for re-producing**_</sub>


