import cv2
from model import load_model, detect
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import numpy as np


def main():
    model = load_model()
    stream(model)


def stream(model, device_index=0):
    category_index = {
        1: {"id": 1, "name": "thumbs_up"},
        2: {"id": 2, "name": "thumbs_down"},
        3: {"id": 3, "name": "hello"},
        4: {"id": 4, "name": "fist"}
    }
    print("Opening window... \nTo exit, press 'q'")
    vid = cv2.VideoCapture(device_index)

    while vid.isOpened(): 
        ret, frame = vid.read()
        image_np = np.array(frame)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        with tf.device("cpu:0"):
            detections = detect(input_tensor, model)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections["detection_boxes"][0].numpy(),
                    np.array(detections["detection_classes"][0], dtype=np.int32)+label_id_offset,
                    detections["detection_scores"][0].numpy(),
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=2,
                    min_score_thresh=.5,
                    agnostic_mode=False)

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            vid.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
