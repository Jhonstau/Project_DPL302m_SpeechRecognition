import librosa
import numpy as np
from sklearn.preprocessing import normalize

class audio_processor:
    def __init__(self, duration=3, max_len=130, n_mfcc=40, n_mels=40):
        self.duration = duration
        self.max_len = max_len
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def __standardize_length(self, y, target_length):
        current_length = len(y)
        if current_length <= target_length:
            pad_length = target_length - current_length
            left = pad_length // 2
            right = pad_length - left
            return np.pad(y, (left, right))
        else:
            start = (current_length - target_length) // 2
            end = start + target_length
            return y[start:end]

    def load_audio(self, file_path):
        y, sr = librosa.load(file_path, sr = 22050)
        target_length = sr * self.duration
        y = self.__standardize_length(y, target_length)
        y = librosa.util.normalize(y)
        return y, sr

    def __pad_or_truncate(self, feat):
        T = feat.shape[1]
        if T < self.max_len:
            return np.pad(feat, ((0, 0), (0, self.max_len - T)), mode='constant')
        else:
            return feat[:, :self.max_len]

    def __standardize_clip(self, feat):
        return normalize(feat, norm='l2', axis=0)

    def __extract_mfcc(self, y, sr):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = self.__pad_or_truncate(mfcc)
        mfcc = self.__standardize_clip(mfcc)
        return mfcc
    
    def __extract_mel(self, y, sr):
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = self.__pad_or_truncate(mel_db)
        mel_db = self.__standardize_clip(mel_db)
        return mel_db
    
    def __extract_features(self, y, sr):
        features = []

        features.append(self.__extract_mfcc(y, sr))
        features.append(self.__extract_mel(y, sr))       

        features = np.stack(features, axis=0)      # (2, 40, T)
        features = features.transpose(0, 2, 1)     # (2, T, 40)
        return features

    def __preprocess(self, file_path):
        y, sr = self.load_audio(file_path)
        features = self.__extract_features(y, sr)
        features = np.expand_dims(features, axis=0)          # (1, 2, T, 40)
        N, D, T, F = features.shape
        features = features.reshape(N, 1, D, T, F)            # (1, 1, 2, T, 40)
        return features

    def batch_preprocess(self, file_paths: list):
        batch = []
        for path in file_paths:
            try:
                feat = self.__preprocess(path)
                batch.append(feat)
            except Exception as e:
                print(f"Error file handling with {path}: {e}")
                continue
        if not batch:
            raise ValueError("Can not handle any file.")
        return np.concatenate(batch, axis=0)