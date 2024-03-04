import numpy as np
import pandas as pd
import os
import nltk
from nltk.stem import PorterStemmer
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')  # Uncomment if not already downloaded

# Fixing the missing closing quote in the directory path
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Fixing indentation and syntax errors in the code block
medicines = pd.read_csv('/kaggle/input/medicine/medicine.csv')
medicines.head()
medicines.shape
medicines.isnull().sum()
medicines.dropna(inplace=True)
medicines.duplicated().sum()
medicines['Description'] = medicines['Description'].apply(lambda x: x.split())
medicines['Reason'] = medicines['Reason'].apply(lambda x: x.split())
medicines['Description'] = medicines['Description'].apply(lambda x: [i.replace(" ", "") for i in x])
medicines['Description'] = medicines['Description'].apply(lambda x: [i.replace("", "") for i in x])  # Corrected this line
medicines['tags'] = medicines['Description'] + medicines['Reason']

new_df = medicines[['index', 'Drug, Name', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Fixing import statement for PorterStemmer
ps = PorterStemmer()

# Fixing import statement for CountVectorizer
cv = CountVectorizer(stop_words='english', max_features=5000)


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


new_df['tags'] = new_df['tags'].apply(stem)

vectors = cv.fit_transform(new_df['tags']).toarray()

# Fixing indentation and syntax errors in the code block
similarity = cosine_similarity(vectors)
print(similarity[1])


def recommended(medicine):
    medicine_index = new_df[new_df['Drug, Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in medicines_list:
        print(new_df.iloc[i[0]]['Drug, Name'])


recommended("MontekLC 75mg Syrup 60mlMontekLC 500mg Tablet 10's")

# Correcting import statement for PorterStemmer
import pickle

# Correcting variable name and fixing syntax errors
pickle.dump(new_df.to_dict(), open('medicine_dict.pkl', 'wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))