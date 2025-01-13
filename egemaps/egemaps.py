import numpy as np
import os
import opensmile
import soundfile as sf

smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                        feature_level=opensmile.FeatureLevel.Functionals)

wav_dir_root='./data'
output_dir = './eGeMAPS'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

task_list = ["0", "1", "2"]

for task in task_list:
    wav_dir = os.path.join(wav_dir_root, f"task-{task}")
    fea_dic = {}
    for audio_file in os.listdir(wav_dir):
        if audio_file.split('.')[-1] != 'wav':
            continue
        ses_id = audio_file.split('-')[0]
        print(ses_id)
        wav_file = os.path.join(wav_dir,audio_file)
        wav, sample_rate = sf.read(wav_file, always_2d=True)
        assert sample_rate == 16000
        y = smile.process_file(wav_file)
        fea_list = []
        for items in y:
            a = y[items]
            fea_list = fea_list+a.values.tolist()
        fea = np.array(fea_list)
        fea_dic[ses_id] = fea
    np.save(os.path.join(output_dir, f'eGeMAPS-{task}.npy'), fea_dic)
