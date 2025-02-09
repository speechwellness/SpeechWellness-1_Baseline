# Baseline for the 1st SpeechWellness Challenge

This repository provides the implementation of the baselines for the 1st SpeechWellness Challenge. Participants can refer to the baseline and associated resources to build and evaluate their models.

## Baseline Result and Reference Paper

The baseline achieves an accuracy of 0.61 on the test set using wav2vec 2.0 + BERT, with the confusion matrix: 
|               | Predicted: No Risk | Predicted: At Risk |
|---------------|--------------------|--------------------|
| Actual: No Risk |  31               |  19               |
| Actual: At Risk |  20                |  30              |
For more details about the challenge data, baseline methods and results, please refer to the [challenge paper](https://arxiv.org/abs/2501.06474).

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

To run the baseline, follow these steps:

1. Reorganize Audio Files: Arrange the audio files by speech task into the following file structure: audio/{task}/{wav_file}
2. Preprocess: Run ```python preprocess/json_prepare.py``` to generate the JSON files required for training, and ```python preprocess/transcribe.py``` for ASR.
3. Train: Run ```bash {audio/text/combine}/train.sh``` to start training

Before running the scripts, ensure that you update the file paths in the code to match the local storage paths on your machine.

## Citation
If you use our dataset, please cite the following paper:
```bibtex
@article{wu20251stspeechwellnesschallengedetecting,
      title={The 1st SpeechWellness Challenge: Detecting Suicidal Risk Among Adolescents}, 
      author={Wen Wu and Ziyun Cui and Chang Lei and Yinan Duan and Diyang Qu and Ji Wu and Bowen Zhou and Runsen Chen and Chao Zhang},
      journal={arXiv preprint arXiv:2501.06474},
      year={2025},
}
```
