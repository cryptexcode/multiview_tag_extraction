# Multi-view Story Characterization from Movie Plot Synopses and Reviews
EMNLP 2020

[Paper](https://www.aclweb.org/anthology/2020.emnlp-main.454.pdf)
[Project Page](https://ritual.uh.edu/multiview-tag-2020/)

## Contributors
- [Sudipta Kar](http://sudiptakar.info)
- [Gustavo Aguilar](https://gustavoaguilar.io/)
- [Mirella Lapata](https://homepages.inf.ed.ac.uk/mlap/)
- [Thamar Solorio](http://solorio.uh.edu)


## Abstract
> This paper considers the problem of characterizing stories by inferring properties such as theme and style using written synopses and reviews of movies. We experiment with a multi-label dataset of movie synopses and a tagset representing various attributes of stories (e.g., genre, type of events). Our proposed multi-view model encodes the synopses and reviews using hierarchical attention and shows improvement over methods that only use synopses. Finally, we demonstrate how we can take advantage of such a model to extract a complementary set of story-attributes from reviews without direct supervision. We have made our dataset and source code publicly available at https://ritual.uh.edu/multiview-tag-2020.



This repository currently contains a streamlit app to run the tag predictor model. We will add more code soon.

## Install requirements and get resources
run `sh init.sh` in terminal


## Run the app to predict
```
cd demo_app
streamlit run app.py

```

## Bibtex
```
@inproceedings{kar-etal-2020-multi,
    title = "Multi-view Story Characterization from Movie Plot Synopses and Reviews",
    author = "Kar, Sudipta  and
      Aguilar, Gustavo  and
      Lapata, Mirella  and
      Solorio, Thamar",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.454",
    doi = "10.18653/v1/2020.emnlp-main.454",
    pages = "5629--5646",
}
```

* For any queries, please contact the first author at skar3 AT uh DOT edu
