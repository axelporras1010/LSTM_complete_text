import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Cargar el contenido del archivo
def cargar_texto(ruta):
    with open(ruta, 'r', encoding='utf-8') as archivo:
        return archivo.read()

texto = cargar_texto('dataset.txt')

# Configurar el tokenizador y contar palabras
def preparar_tokenizador(texto):
    tokenizador = Tokenizer()
    tokenizador.fit_on_texts([texto])
    return tokenizador

tokenizador = preparar_tokenizador(texto)
vocab_size = len(tokenizador.word_index) + 1

# Crear secuencias para entrenamiento
def generar_secuencias(texto, tokenizador):
    secuencias = []
    for i in range(1, len(texto)):
        secuencia = texto[:i + 1]
        secuencias.append(secuencia)
    return pad_sequences(tokenizador.texts_to_sequences(secuencias))

secuencias = generar_secuencias(texto, tokenizador)
X, y = secuencias[:, :-1], secuencias[:, -1]

# Definir y compilar el modelo
def construir_modelo(vocab_size, input_length):
    modelo = tf.keras.Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=input_length),
        LSTM(100),
        Dense(vocab_size, activation='softmax')
    ])
    modelo.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return modelo

modelo = construir_modelo(vocab_size, X.shape[1])

# Entrenar el modelo
modelo.fit(X, y, epochs=20, verbose=1)

# Generar texto a partir de un texto inicial
def generar_texto(modelo, tokenizador, texto_inicial, num_palabras, longitud_max):
    for _ in range(num_palabras):
        secuencia = tokenizador.texts_to_sequences([texto_inicial])[0]
        secuencia = pad_sequences([secuencia], maxlen=longitud_max, padding='pre')
        prediccion = np.argmax(modelo.predict(secuencia, verbose=0))
        palabra_predicha = tokenizador.index_word.get(prediccion, '')
        texto_inicial += ' ' + palabra_predicha
    return texto_inicial

texto_generado = generar_texto(modelo, tokenizador, "El sol ", 5, X.shape[1])
print(texto_generado)