# Load the LDA model from sk-learn
import os

from sklearn.decomposition import LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

cwd = os.getcwd()
mallet_path = os.path.join(cwd, "mallet-2.0.8/bin/mallet")
corpus_path = '/content/file.txt'

def generate_word_cloud(pd_series):
    # Convert to list
    long_string = ",".join(list(pd_series.values))
    # Create a WordCloud object
    wordcloud = WordCloud(
        background_color="white",
        width=1000,
        height=250,
        max_words=300,
        contour_width=3,
        contour_color="steelblue",
        min_word_length=3,
    )
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    return wordcloud

# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic #{topic_idx+1:d}:")
        print(" ".join([words[i]
        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
        
def generate_lda_models(number_topics, count_data):
    # Create and fit the LDA model
    model_list = []
    model_perplexity = []
    for num in number_topics:
        lda = LDA(n_components=num, max_iter=100, random_state=100, n_jobs=-1)
        lda.fit(count_data)
        model_list.append(lda)
        model_perplexity.append(lda.perplexity(count_data))

    return model_list, model_perplexity


def plot_topic_word_cloud(lda_model, count_vectorizer, fig_size):
    topic_words = []
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda_model.components_):
        topic_words.append(
            " ".join([words[i] for i in topic.argsort()[: -10 - 1 : -1]])
        )
    fig, axes = plt.subplots(
        fig_size[0], fig_size[1], figsize=(5, 5), sharex=True, sharey=True, dpi=250
    )

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        #     topic_words = dict(topics[i][1])
        #     cloud.generate_from_frequencies(topic_words, max_font_size=300)
        wordcloud = WordCloud(
            background_color="white",
            width=1000,
            height=500,
            max_words=300,
            contour_width=3,
            contour_color="steelblue",
            min_word_length=3,
        )
        wordcloud.generate(topic_words[i])
        # Visualize the word cloud
        #     wordcloud.to_image()
        plt.gca().imshow(wordcloud)
        plt.gca().set_title("Topic " + str(i + 1), fontdict=dict(size=12))
        plt.gca().axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

def ldaMalletConvertToldaGen(mallet_model):
    model_gensim = gensim.models.ldamodel.LdaModel(
        id2word=mallet_model.id2word, 
        num_topics=mallet_model.num_topics, 
        alpha=mallet_model.alpha, 
        eta=0, 
        iterations=100, 
        gamma_threshold=0.001, 
        random_state=100, 
        dtype=np.float32
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim
        
        
def compute_coherence_values(dictionary, corpus, texts, canditates):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in canditates:
        model = gensim.models.wrappers.ldamallet.LdaMallet(
            mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary, iterations=100, random_seed=100
        )
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
