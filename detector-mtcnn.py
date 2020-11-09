from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = cv2.cvtColor(cv2.imread("face.png"), cv2.COLOR_BGR2RGB)

detector = MTCNN()
result = detector.detect_faces(image)

print(result[0]['box'])
box = result[0]['box']
cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2],box[1]+box[3]), (0, 255, 0), 5)

imgplot = plt.imshow(image)
plt.show()

