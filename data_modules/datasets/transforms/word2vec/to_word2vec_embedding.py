import numpy as np

from gensim.models import Word2Vec


class ToWord2vecEmbedding:
    def __init__(self, word2vec_model_path: str, key_to_encode: str, window_size: int = 3, use_norm: bool = True):
        self.model = Word2Vec.load(word2vec_model_path)
        self.key_to_encode = key_to_encode
        self.window_size = window_size
        self.use_norm = use_norm

    def __call__(self, sample):
        sequence = sample[self.key_to_encode][0]
        divided = np.array([sequence[i:i + self.window_size] for i in range(0, len(sequence), self.window_size)])
        sample[self.key_to_encode] = np.expand_dims(
            np.array([self.model.wv.get_vector(word, norm=self.use_norm) for word in divided]), axis=0)
        return sample
