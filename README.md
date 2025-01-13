# Baseline for the 1st SpeechWellness Challenge

This repository provides the implementation of the baselines for the 1st SpeechWellness Challenge. Participants can refer to the baseline and associated resources to build and evaluate their models.

## Reference Paper

For more details about the challenge data, baseline methods and results, please refer to the paper (arxiv url)

## Test Result Submission

To submit test results, please organize your predictions in a JSON file formatted as a dictionary with the structure: {id: prediction}. See the [sample.json](https://github.com/speechwellness/SpeechWellness-1_Baseline/blob/main/sample.json) for example.

## Code Usage

The code is developed with Python 3.10. Install dependencies using:
```bash
pip install -r requirements.txt
```

The file structure of this repository is:
```plaintext
├── audio                       # Scripts for W2V2 model finetuning and feature extracting
├── text                        # Scripts for BERT model finetuning and feature extracting
├── combine                     # Scripts for classify training
├── egemaps                     # Scripts for eGeMAPS extracting and SVM classification
├── preprocess                  
│   ├── json_prepare.py         # csv -> json
│   └── transcribe.py           # ASR
├── metric_compute.py           # Calculate metrics
├── soft_vote.py                # Soft vote for W2V2+BERT
├── vote.py                     # Have vote for eGeMAPS+SVM
├── sample.json                 # An example for test result submission
└── requirements.txt            # Dependencies
```

## Citation
If you use our dataset, please cite the following paper:
```bibtex
@article{
...
}
```
