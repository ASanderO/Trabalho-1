import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
img = cv2.imread('pessoa.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Erro: Imagem não encontrada.")
    exit()

# Calcular e mostrar o histograma
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure(figsize=(10, 5))
plt.plot(hist)
plt.title('Histograma da imagem')
plt.xlabel('Intensidade de pixel')
plt.ylabel('Frequência')
plt.show(block=False)  # Não bloqueia a execução do programa

# Equalização do histograma
img_eq = cv2.equalizeHist(img)
cv2.imshow('Imagem Original', img)
cv2.imshow('Imagem Equalizada', img_eq)

# Suavização por média
blur_img = cv2.blur(img, (5, 5))
cv2.imshow('Imagem Suavizada com Média', blur_img)

# Suavização Gaussiana
gauss_img = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow('Imagem Suavizada com Gaussiana', gauss_img)

# Suavização pela Mediana
median_img = cv2.medianBlur(img, 5)
cv2.imshow('Imagem Suavizada com Mediana', median_img)

# Filtro Bilateral
bilateral_img = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow('Imagem Suavizada com Filtro Bilateral', bilateral_img)

# Carregar o classificador Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detectar faces na imagem
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# Desenhar retângulos ao redor de cada face e rotular
labeled_img = np.copy(img)
for (x, y, w, h) in faces:
    cv2.rectangle(labeled_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(labeled_img, 'Person 100%', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imshow('Imagem com Detecção de Faces', labeled_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
