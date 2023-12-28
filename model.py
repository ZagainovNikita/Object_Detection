from object_detection.utils import config_util
from object_detection.builders import model_builder
import tensorflow as tf

def load_model():
    print("Loading model from checkpoint...")

    config = config_util.get_configs_from_pipeline_file(
        "mobile_net\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\pipeline.config"
    )
    model_config = config["model"]
    model_config.ssd.num_classes = 4
    model_config.ssd.freeze_batchnorm = True

    model = model_builder.build(model_config, False)

    ckpt_trained = tf.train.Checkpoint(model=model)
    ckpt_trained.restore(r"mobile_net\trained_model\checkpoint-1")

    print("The model is loaded successfully")

    return model

@tf.function
def detect(input_tensor, model):
    preprocessed_image, shapes = model.preprocess(input_tensor)
    prediction_dict = model.predict(preprocessed_image, shapes)
    return model.postprocess(prediction_dict, shapes)