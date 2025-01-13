import os
import json
from funasr import AutoModel

wav_root = "./data"
transcribe_root = "./transcriptions"
model = AutoModel(
    model="paraformer-zh", model_revision="v2.0.4",
    vad_model="fsmn-vad", vad_model_revision="v2.0.4",
    punc_model="ct-punc-c", punc_model_revision="v2.0.4",
    device="cuda"
)

for sub in os.listdir(wav_root):
    sub_dir = os.path.join(wav_root, sub)
    if not os.path.isdir(sub_dir):
        continue
    for wav_name in os.listdir(sub_dir):
        result_save = []
        if not (wav_name.endswith('.wav') or wav_name.endswith('.WAV')):
            continue
        save_dir = os.path.join(transcribe_root, sub)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(transcribe_root, sub, wav_name.lower().replace('.wav', '.json'))
        if os.path.isfile(save_path):
            continue
        wav_file = os.path.join(sub_dir, wav_name)
        print(wav_file)

        try:
            res = model.generate(input=wav_file, sentence_timestamp=True)[0]
            for sentence in res["sentence_info"]:
                result_save.append(
                    {
                        "start": sentence["timestamp"][0][0],
                        "end": sentence["timestamp"][-1][-1],
                        "text": sentence["text"]
                    }
                )
            with open(save_path, 'w') as f:
                json.dump(result_save, f, ensure_ascii=False, indent=4)
        except Exception as e:
            with open(os.path.join(transcribe_root, "fail_list.txt"), 'a+') as f:
                f.write(wav_name + '\n')
