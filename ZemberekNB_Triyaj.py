import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from zemberek import TurkishMorphology


morphology = TurkishMorphology.create_with_defaults()

def preprocess_turkish(text):
    if pd.isna(text): return ""
    tokens = str(text).lower().split()
    lemmatized = []
    for token in tokens:
        results = morphology.analyze(token)
        found_lemma = False
        for res in results:
            if res.get_stem():
                lemmatized.append(res.get_stem())
                found_lemma = True
                break
        if not found_lemma:
            lemmatized.append(token)
    return " ".join(lemmatized)


data = pd.read_csv("temizlenmisdataset.csv")

data['clean_text'] = data['text'].apply(preprocess_turkish)

X = data['clean_text']
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


vectorizer = TfidfVectorizer(
    ngram_range=(1, 3), 
    max_features=2000, 
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)


while True:
    user_input = input("\nHasta şikayetini giriniz: ")
    
    if user_input.lower() == 'q':
        print("Sistemden çıkılıyor...")
        break
    
    if not user_input.strip():
        continue

    
    processed = preprocess_turkish(user_input)
    
    
    vec_input = vectorizer.transform([processed])
    prediction = model.predict(vec_input)[0]
    probabilities = model.predict_proba(vec_input)[0]
    classes = model.classes_.tolist()

    
    print(f"\n>>> ANALİZ SONUCU <<<")
    print(f"Orijinal Metin : {user_input}")
    print(f"Zemberek Kök   : {processed}")
    print(f"KARAR          : {prediction.upper()}")
    print("-" * 25)
    for i, cls in enumerate(classes):
        print(f"Olasılık %{cls:<12}: {probabilities[i]*100:.2f}")


y_pred = model.predict(X_test_vec)
print("\nGENEL MODEL PERFORMANSI:")
print(classification_report(y_test, y_pred))
