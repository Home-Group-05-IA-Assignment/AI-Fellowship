# IA-Fellowship
One App of recognition of emotion and stress. 

El notebook de la limpieza de datos esta adjunto y los outputs también para su procesamiento
Los archivos de la carpeta 'clean-data' se relacionan así: 
- abbreviations.json: conjunto de abreviaciones del inglés para manejarlas.
- hg-05-ia.ipynb: archivo jupyter con proceso de limpieza de datos. 
  - Se eliminan nulos y datos reptidos.
  - Se liberan espacios extras.
  - Se maneja simbolos y puntos para retirarlos.
  - Se coloca todo el text en minúscula.
  - Se maneja las abreviaciones del inglés.
  - Se crea columna con nombre de etiquetas para valore númericos sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5)
  - Se crea columna del tamaño del texto. 
  - Se eliminan emojis
  - Se eliminan 'stop words', palabras sin significado.
  - Se crea columna para 'Text_Stemmed' con palabras de texto derivado para simplificar.
- text.csv: datos sin procesar.
- cleaned_data: datos procesados y limpios sin dividir.
- X_test.pkl: datos segmentado para test.
- X_train.pkl: datos segmentados para entrenamiento.
- y_test.pkl: etiquetas segmentadas para test en caso de usar cierta implementación de BERT.
- y_train.pkl: etiquetas segmentadas para entrenamient en caso de usar cierta implementación de BERT.

Enlace al google [Colab](https://colab.research.google.com/drive/1Fbrna_8GAClKRJUhp9pRzK2Sd5XwCetx?usp=sharing)

No se tokerizan ni vectorizan pues esto puede afectar la implementación de BERT. Depende de que biblioteca se use.