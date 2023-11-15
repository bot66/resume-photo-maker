import onnxruntime
import cv2
import numpy as np
import argparse

# The common resume photo size is 35mmx45mm
RESUME_PHOTO_W = 350
RESUME_PHOTO_H = 450


# modified from https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/yunet.py
class YuNet:
    def __init__(
        self,
        modelPath,
        inputSize=[320, 320],
        confThreshold=0.6,
        nmsThreshold=0.3,
        topK=5000,
        backendId=0,
        targetId=0,
    ):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize)  # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId,
        )

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId,
        )

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        # Forward
        faces = self._model.detect(image)
        return faces[1]


class ONNXModel:
    def __init__(self, model_path, input_w, input_h):
        self.model = onnxruntime.InferenceSession(model_path)
        self.input_w = input_w
        self.input_h = input_h

    def preprocess(self, rgb, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        # convert the input data into the float32 input
        img_data = (
            np.array(cv2.resize(rgb, (self.input_w, self.input_h)))
            .transpose(2, 0, 1)
            .astype("float32")
        )

        # normalize
        norm_img_data = np.zeros(img_data.shape).astype("float32")

        for i in range(img_data.shape[0]):
            norm_img_data[i, :, :] = img_data[i, :, :] / 255
            norm_img_data[i, :, :] = (norm_img_data[i, :, :] - mean[i]) / std[i]

        # add batch channel
        norm_img_data = norm_img_data.reshape(1, 3, self.input_h, self.input_w).astype(
            "float32"
        )
        return norm_img_data

    def forward(self, image):
        input_data = self.preprocess(image)
        output_data = self.model.run(["argmax_0.tmp_0"], {"x": input_data})

        return output_data


def parse_args():
    parser = argparse.ArgumentParser(description="Resume Photo Maker")
    parser.add_argument(
        "--background_color",
        "-bg",
        nargs="+",
        type=int,
        default=(255, 255, 255),
        help="Set the background color RGB values.",
    )
    parser.add_argument(
        "--image", "-i", type=str, default="images/elon.jpg", help="Input image path."
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    bgr = cv2.imread(args.image)
    h, w, _ = bgr.shape

    # Initialize models
    face_detector = YuNet("models/face_detection_yunet_2023mar.onnx")
    face_detector.setInputSize([w, h])
    human_segmentor = ONNXModel(
        "models/human_pp_humansegv2_lite_192x192_inference_model.onnx", 192, 192
    )

    # yunet uses opencv bgr image format
    detections = face_detector.infer(bgr)

    for idx, det in enumerate(detections):
        # bounding box
        pt1 = np.array((det[0], det[1]))
        pt2 = np.array((det[0] + det[2], det[1] + det[3]))

        # face landmarks
        landmarks = det[4:14].reshape((5, 2))
        right_eye = landmarks[0]
        left_eye = landmarks[1]

        angle = np.arctan2(right_eye[1] - left_eye[1], (right_eye[0] - left_eye[0]))
        rmat = cv2.getRotationMatrix2D((0, 0), -angle, 1)

        # apply rotation
        rotated_bgr = cv2.warpAffine(bgr, rmat, (bgr.shape[1], bgr.shape[0]))
        rotated_pt1 = rmat[:, :-1] @ pt1
        rotated_pt2 = rmat[:, :-1] @ pt2

        face_w, face_h = rotated_pt2 - rotated_pt1
        up_length = int(face_h / 4)
        down_length = int(face_h / 3)
        crop_h = face_h + up_length + down_length
        crop_w = int(crop_h * (RESUME_PHOTO_W / RESUME_PHOTO_H))

        pt1 = np.array(
            (rotated_pt1[0] - (crop_w - face_w) / 2, rotated_pt1[1] - up_length)
        ).astype(np.int32)
        pt2 = np.array((pt1[0] + crop_w, pt1[1] + crop_h)).astype(np.int32)

        resume_photo = rotated_bgr[pt1[1] : pt2[1], pt1[0] : pt2[0], :]

        rgb = cv2.cvtColor(resume_photo, cv2.COLOR_BGR2RGB)
        mask = human_segmentor.forward(rgb)
        mask = mask[0].transpose(1, 2, 0)
        mask = cv2.resize(
            mask.astype(np.uint8), (resume_photo.shape[1], resume_photo.shape[0])
        )

        resume_photo[mask == 0] = args.background_color

        resume_photo = cv2.resize(resume_photo, (RESUME_PHOTO_W, RESUME_PHOTO_H))
        cv2.imwrite(f"masked_resume_photo_{idx}.jpg", resume_photo)
