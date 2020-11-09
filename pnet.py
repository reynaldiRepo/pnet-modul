# reference https://github.com/ipazc/mtcnn.git

from mtcnn_.network.factory import NetworkFactory
import cv2
import numpy as np
import pkg_resources

class pnet(NetworkFactory):
    def __init__(self, image ,weights_file, scale_factor: float = 0.709, min_face_size: int = 20):
        # param: image cv2.cvtColor
        # param: scale factor is float
        # param: min face size is float
        # param: scale factor is float
        self.image = image;
        self.weights_file =  np.load(weights_file, allow_pickle=True).tolist()
        self.p_net = self.build_pnet()
        self.p_net.set_weights(self.weights_file['pnet'])
        self._scale_factor = scale_factor;
        self._min_face_size = min_face_size
        
        # get image shape
        height, width, _ = self.image.shape
        # scale image to image pyramid 12x scaling
        m = 12 / self._min_face_size
        min_layer = np.amin([height, width]) * m
        scales = self.__compute_scale_pyramid(m, min_layer)
        # print(scales);

        total_boxes = np.empty((0, 9))
        print(total_boxes)

        for scale in scales:
            scaled_image = self.__scale_image(self.image, scales)
            print(scaled_image);
            break;
    
    def __compute_scale_pyramid(self, m, min_layer):
        scales = []
        factor_count = 0
        while min_layer >= 12:
            scales += [m * np.power(self._scale_factor, factor_count)]
            min_layer = min_layer * self._scale_factor
            factor_count += 1
        return scales


    def __scale_image(self, image, scale: float):
        height, width, _ = image.shape

        width_scaled = int(np.ceil(width * scale))
        height_scaled = int(np.ceil(height * scale))

        im_data = cv2.resize(image, (width_scaled, height_scaled), interpolation=cv2.INTER_AREA)

        # Normalize the image's pixels
        im_data_normalized = (im_data - 127.5) * 0.0078125

        return im_data_normalized

    def __generate_bounding_box(imap, reg, scale, t):
    
        # use heatmap to generate bounding boxes
        stride = 2
        cellsize = 12

        imap = np.transpose(imap)
        dx1 = np.transpose(reg[:, :, 0])
        dy1 = np.transpose(reg[:, :, 1])
        dx2 = np.transpose(reg[:, :, 2])
        dy2 = np.transpose(reg[:, :, 3])

        y, x = np.where(imap >= t)

        if y.shape[0] == 1:
            dx1 = np.flipud(dx1)
            dy1 = np.flipud(dy1)
            dx2 = np.flipud(dx2)
            dy2 = np.flipud(dy2)

        score = imap[(y, x)]
        reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))

        if reg.size == 0:
            reg = np.empty(shape=(0, 3))

        bb = np.transpose(np.vstack([y, x]))

        q1 = np.fix((stride * bb + 1) / scale)
        q2 = np.fix((stride * bb + cellsize) / scale)
        boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])

        return boundingbox, reg

    def __nms(boxes, threshold, method):
        """
        Non Maximum Suppression.

        :param boxes: np array with bounding boxes.
        :param threshold:
        :param method: NMS method to apply. Available values ('Min', 'Union')
        :return:
        """
        if boxes.size == 0:
            return np.empty((0, 3))

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        sorted_s = np.argsort(s)

        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while sorted_s.size > 0:
            i = sorted_s[-1]
            pick[counter] = i
            counter += 1
            idx = sorted_s[0:-1]

            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h

            if method == 'Min':
                o = inter / np.minimum(area[i], area[idx])
            else:
                o = inter / (area[i] + area[idx] - inter)

            sorted_s = sorted_s[np.where(o <= threshold)]

        pick = pick[0:counter]

        return pick

    def __pad(total_boxes, w, h):
        # compute the padding coordinates (pad the bounding boxes to square)
        tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
        tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
        numbox = total_boxes.shape[0]

        dx = np.ones(numbox, dtype=np.int32)
        dy = np.ones(numbox, dtype=np.int32)
        edx = tmpw.copy().astype(np.int32)
        edy = tmph.copy().astype(np.int32)

        x = total_boxes[:, 0].copy().astype(np.int32)
        y = total_boxes[:, 1].copy().astype(np.int32)
        ex = total_boxes[:, 2].copy().astype(np.int32)
        ey = total_boxes[:, 3].copy().astype(np.int32)

        tmp = np.where(ex > w)
        edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
        ex[tmp] = w

        tmp = np.where(ey > h)
        edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
        ey[tmp] = h

        tmp = np.where(x < 1)
        dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
        x[tmp] = 1

        tmp = np.where(y < 1)
        dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
        y[tmp] = 1

        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

    


if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread("face.png"), cv2.COLOR_BGR2RGB)
    weights_file = pkg_resources.resource_stream('mtcnn_', 'data/mtcnn_weights.npy')
    stage1 = pnet(image ,weights_file);