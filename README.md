# Reliability-Aware-Tester-based-evaluation-framework

This is the code repository for the evaluation framework called Reliability Aware Tester based evaluation framework (RATE) for evaluating reliability of user simulators.

**Description:** Reliability Aware Tester based evaluation (RATE) framework is an evaluation framework to compute reliability of user simulators in evaluating interactive retrieval systems. In RATE, we construct evaluation modules called "Tester"'s and then evaluate a set of simulators using a set of testers through iterative reliability propagation.


### Paper

Labhishetty et al., In ECIR 2022, _RATE: A Reliability-Aware Tester-Based Evaluation Framework of User Simulators_, https://dl.acm.org/doi/abs/10.1007/978-3-030-99736-6_23

Labhishetty et al., In SIGIR 2021, _An Exploration of Tester-based Evaluation of User Simulators for Comparing Interactive Retrieval Systems_, https://dl.acm.org/doi/10.1145/3404835.3463091


### Contents

`RATE_simulators_code/` : for building a set of representative simulators to interact with IIR systems to generate sessions. 

`iir_methods/` : code of search interfaces with IIR algorithms, BM25variations, click history based methods, query history based methods.


## Research Usage

If you use our work in your research please cite:

```
@inproceedings{10.1007/978-3-030-99736-6_23,
author = {Labhishetty, Sahiti and Zhai, ChengXiang},
title = {RATE: A Reliability-Aware Tester-Based Evaluation Framework Of User Simulators},
year = {2022},
isbn = {978-3-030-99735-9},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-030-99736-6_23},
doi = {10.1007/978-3-030-99736-6_23},
booktitle = {Advances in Information Retrieval: 44th European Conference on IR Research, ECIR 2022, Stavanger, Norway, April 10–14, 2022, Proceedings, Part I},
pages = {336–350},
numpages = {15},
keywords = {IIR Systems, Reliability of User Simulator, Tester},
location = {Stavanger, Norway}
}

@inproceedings{10.1145/3404835.3463091,
author = {Labhishetty, Sahiti and Zhai, Chengxiang},
title = {An Exploration of Tester-Based Evaluation of User Simulators for Comparing Interactive Retrieval Systems.},
year = {2021},
isbn = {9781450380379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3404835.3463091},
doi = {10.1145/3404835.3463091},
booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1598–1602},
numpages = {5},
keywords = {user simulation evaluation, user simulation, interactive IR systems},
location = {Virtual Event, Canada},
series = {SIGIR '21}
}

```

## License

By using this source code you agree to the license described in https://github.com/sahitilucky/Reliability-Aware-Tester-based-evaluation-framework/blob/master/LICENSE




