import pandas as pd

import pandas as pd
data1 = pd.read_csv("C:\\Users\\PC\Desktop\\Tensorflow\\MYC-main\\MYC-main\\DATA_CLEANED.csv")
# Renommer les colonnes
data1.rename(columns={'sentence': 'tweet', 'polarity': 'label'}, inplace=True)
data1['label'] = data1['label'].replace({-1: 0})


data2 = pd.read_csv("C:\\Users\\PC\\Desktop\\Tensorflow\\data.csv")
data2['label'] = data2['label'].replace({'pos': 1, 'neg': 0})


# Ajouter les données de data2 à data1
data = pd.concat([data1, data2], ignore_index=True)

# Affichage du résultat
print(data)
