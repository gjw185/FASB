# FASB

## Abstact
 >We propose the Flexible Activation Steering with Backtracking (FASB) framework, which dy
namically determines both the necessity and strength of intervention by tracking
 the internal states of the LLMs during generation, considering both the question
 and the generated content. Since intervening after detecting a deviation from the
 desired behavior is often too late, we further propose the backtracking mechanism
 to correct the deviated tokens and steer the LLMs toward the desired behavior.

## Installation
```commandline
conda env create -f environment.yml
conda activate FASB
mkdir -p validation/results_dump/answer_dump
mkdir -p validation/results_dump/summary_dump
mkdir -p validation/splits
mkdir features
git clone https://github.com/sylinrl/TruthfulQA.git
```

## Workflow
(1) Get activations by running bash get_activations.sh.

(2) Get into validation folder and run run_FASB.sh 