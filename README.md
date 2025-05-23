<div align="center">
<p align="center">
  <img src="./assets/model.png" width="75%" height="55%" />
</p>
</div>

<div align="center">
<h1>StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization
</h1>
</div>

We will release our code and weights in a few days, stay tuned!

# Introduction
StepSearch is a method specifically tailored for multi-hop question answering in search-based scenarios. It models the reasoning process as a sequence of tool invocations and information retrieval steps, where each step aims to gather relevant evidence toward answering a complex question. The system is trained using reinforcement learning, particularly the Proximal Policy Optimization (PPO) algorithm with token-level and rewards, to optimize a policy that decides search tools to invoke and what queries to issue at each stage.

- **Universal multi-hop search data.** We develop a novel MuSiQue-based pipeline, contributing 60k filtered sub-question search keywords that generalize across retrieval datasets.
- **StepSearch: Step-wise RL with dual rewards.** We augment PPO with token-level rewards—information gain and redundancy penalties—for both query formulation and document retrieval.
- **State-of-the-art performance.** StepSearch outperforms standard RL baselines by *5.7%*, *9.1%*, *10.0%*, and *15.2%* absolutely on diverse multi-hop Q&A benchmarks.

# News
- **[2025.05.20]** Released the initial paper, model weights and dataset.

#  Performance

###  Main Results

<div align="center">
    <img src="./assets/main_results.png" width="69%" height="auto" />
</div>

###  Different RL Comparison 

<div align="center">
    <img src="./assets/RL_comparision_table.png" width="50%" height="auto" />
</div>
<div align="center">
    <img src="./assets/RL-compare-answer-f1.png" width="40%" height="auto" />
    <img src="./assets/RL-compare-response-len.png" width="40%" height="auto" />
</div>

###  Ablation Study

<div align="center">
    <img src="./assets/ablation_table.png" width="50%" height="auto" />
</div>
<div align="center">
    <img src="./assets/AB-ss-answer-f1.png" width="40%" height="auto" />
    <img src="./assets/AB-ss-response-len.png" width="40%" height="auto" />
    <img src="./assets/AB-skr.png" width="80%" height="auto" />
</div>

###  Case Study

<div align="center">
    <img src="./assets/case_study_1.png" width="70%" height="auto" />
    <img src="./assets/case_study_2.png" width="70%" height="auto" />
</div>


#  Acknowledgements

This work is implemented based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [veRL](https://github.com/volcengine/verl), and [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main). We sincerely thank the authors of these projects for their valuable contributions to the open-source community.


# Citation

If this work is helpful, please kindly cite as:

```bigquery
@misc{wang2025stepsearchignitingllmssearch,
      title={StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization}, 
      author={Ziliang Wang and Xuhui Zheng and Kang An and Cijun Ouyang and Jialu Cai and Yuhang Wang and Yichao Wu},
      year={2025},
      eprint={2505.15107},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.15107}, 
}
```
