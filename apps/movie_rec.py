import nltk
nltk.download('stopwords')
from utils import *


def write():
    """Writes content to the app"""
    st.title("Natural Language processing (NLP)\n")

    box = pd.read_csv('data/box_office_samp20k.csv')  

    st.header("Movie recommendation 🎞️")
    st.subheader("Find movie by title")
    title = st.text_input('Insert Movie title', 'Harry Potter')
    if st.button('Search'):
      st.write(f"**Result of your search**")
      st.write(search(box, title))

    st.subheader("Find similar movie")
    movie_id = st.number_input('Insert Movie id', value = 54001)
    st.write(find_similar_movies(box, movie_id))

    st.header("Sentiment analysis 😐 ☹️ 🙂")
    sentiment = SentimentIntensityAnalyzer()
    text_1 = "The book was a perfect balance between wrtiting style and plot."
    text = st.text_input('Insert your text', text_1)
    sent_1 = sentiment.polarity_scores(text)
    res = max(sent_1.items(), key=lambda k: k[1])
    st.write(f"**Sentiment of your text is :** {res[0]} with a score of {res[1]}")

    st.header("Frequencies of Words (Most common 4)")
    it_string = ''' L'azione si svolge nella bella Verona,
                  dove fra due famiglie di uguale nobiltà,
                  per antico .odio nasce una nuova discordia
                  che sporca di sangue le mani dei cittadini.
                  Da questi nemici discendono i due amanti,
                  che, nati sotto contraria stella, dopo pietose vicende, con la loro
                  morte, annientarono l'odio di parte.
                  Le tremende lotte del loro amore,
                  già segnato dalla morte, l'ira spietata dei genitori,
                  che ha fine soltanto con la morte dei figli,
                  ecco quello che la nostra scena vi offrirà in due ore.
                  Se ascolterete con pazienza, la nostra fatica
                  cercherà di compensare qualche mancanza.'''
    
    input_txt = st.text_area('Text to analyze', it_string)
    
    useless_words = stopwords.words("italian") + list(string.punctuation)
    filtered_words = [word for word in word_tokenize(input_txt) if not word in useless_words]
    most_frequent_words = FreqDist(filtered_words).most_common(4)
    st.write("**The frequencies of your text is :**\n")
    st.text(most_frequent_words)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(*zip(*most_frequent_words), color = 'lightblue')
    plt.xlabel("Freq")
    st.pyplot(fig)


if __name__ == "__main__":
    write()      
