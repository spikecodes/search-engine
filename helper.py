import spacy

def lemma(texts):
    # Load the spaCy English model
    nlp = spacy.load('en_core_web_sm')
    # Lemma
    doc = nlp(texts)

    # Extract lemmatized tokens
    lemmatized_tokens = [token.lemma_ for token in doc]

    # Join the lemmatized tokens into a sentence
    return(' '.join(lemmatized_tokens))