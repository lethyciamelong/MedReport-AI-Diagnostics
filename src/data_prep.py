import pandas as pd

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. On simplifie les colonnes
    df = df[['Image Index', 'Finding Labels', 'Patient ID', 'Patient Age', 'Patient Gender']]
    
    # 2. Création des colonnes pour chaque pathologie
    # On récupère toutes les maladies uniques
    all_labels = set()
    df['Finding Labels'].str.split('|').apply(all_labels.update)
    all_labels = sorted(list(all_labels))
    
    for label in all_labels:
        if label != 'No Finding':
            df[label] = df['Finding Labels'].map(lambda x: 1 if label in x else 0)
            
    return df, all_labels

if __name__ == "__main__":
    # Test local
    data, labels = load_and_clean_data('../data/Data_Entry_2017.csv')
    print(f"Nombre de patients uniques : {data['Patient ID'].nunique()}")
    print(f"Maladies à prédire : {labels}")