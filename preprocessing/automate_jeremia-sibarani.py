from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump

def preprocess_data(data, target_column, processor_save_path, clean_train_dataset_save_path, clean_test_dataset_save_path):



  # Memilih fitur numerik, dengan mengecualikan kolom target
  numeric_features = data.drop(target_column, axis=1).select_dtypes(include='number').columns.tolist()
  
  # Memilih fitur kategorik, dengan mengecualikan kolom target
  categoric_features = data.drop(target_column, axis=1).select_dtypes(include='object').columns.tolist()




  # Transformer untuk fitur numerik
  numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
  ])

  # Transformer untuk fitur kategorik
  categoric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('label_encoder', OneHotEncoder(handle_unknown='ignore'))
  ])

  # Membangun preprocessing pipeline
  preprocessor = ColumnTransformer(
    transformers=[
      ('num', numeric_transformer, numeric_features),
      ('cat', categoric_transformer, categoric_features)
    ]
  )

  # Memisahkan fitur dan target
  X = data.drop(target_column, axis=1)
  y = data[target_column]

  # Splitting data dengan persentase 70:30
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # Terapkan preprocessing data menggunakan pipeline preprocessor
  X_train = preprocessor.fit_transform(X_train)
  X_test = preprocessor.transform(X_test)

  # Mengambil nama kolom dari processor
  column_names = preprocessor.get_feature_names_out()
  
  # Membentuk dataframe untuk data latih
  train_df = pd.DataFrame(X_train, columns=column_names)
  train_df[target_column] = y_train
  # Membentuk dataframe untuk data uji
  test_df = pd.DataFrame(X_test, columns=column_names)
  test_df[target_column] = y_test

  # Simpan data yang telah bersih ke dalam format csv
  train_df.to_csv(clean_train_dataset_save_path)
  test_df.to_csv(clean_test_dataset_save_path)

  # Simpan preprocessing pipeline
  dump(preprocessor, processor_save_path)




# Definisikan path untuk sumber dataset, simpan dataset yang telah bersih, dan simpan preprocessing pipeline
raw_dataset_path = '../raw_data/churn.csv'
clean_train_path = './clean_dataset/clean_train.csv'
clean_test_path = './clean_dataset/clean_test.csv'
preprocessing_pipeline_path = './pipeline/preprocessing_pipeline.joblib'


# Membaca raw data
df = pd.read_csv(raw_dataset_path)
# Membuang kolom identifier
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Menjalankan automasi pipeline
preprocess_data(df, 'Exited', preprocessing_pipeline_path, clean_train_path, clean_test_path)