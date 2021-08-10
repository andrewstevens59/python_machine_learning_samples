from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

path = get_tmpfile("word2vec.model")

model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)

model.save("word2vec.model")
model.train([["hello", "world"]], total_examples=1, epochs=1)

print (model.wv['computer'])