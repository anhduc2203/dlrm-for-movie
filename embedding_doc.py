import gensim
import pandas as pd


path = "/home/ducva/PythonProjects/dlrm/recommender_pytorch-master/Data/u.item.plot"
df = pd.read_csv(path, index_col=None)
plot = []
for i in range(df.shape[0]):
    plot.append(df['plot_1'][i])

# Create the tagged document needed for Doc2Vec
def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


train_data = list(create_tagged_document(plot))

# Init the Doc2Vec model
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=0, epochs=40)

# Build the Volabulary
model.build_vocab(train_data)

# Train the Doc2Vec model
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

s = "cowboy doll profoundly threatened jealous new spaceman"

#print(model.infer_vector(s.split()))
print(train_data[:1])
