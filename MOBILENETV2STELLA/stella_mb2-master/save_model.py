import logging
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

from object_detection.utils import config_util
from object_detection.builders import model_builder

config_file = 'network/ssd/pipeline.config'
ckpt_dir = 'train'

configs = config_util.get_configs_from_pipeline_file(config_file)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()

input_shape = (1, model_config.ssd.image_resizer.fixed_shape_resizer.height, model_config.ssd.image_resizer.fixed_shape_resizer.width, 3)
print(input_shape)
input_example = tf.random.normal(input_shape)
_ = detection_model(input_example)


@tf.function(input_signature=[tf.TensorSpec(shape=[1, 160,160, 3], dtype=tf.float32, name='input_tensor')])
def serving_default(input_tensor):
    detections = detection_model(input_tensor, training=False)
    return detections

# Create a concrete function for serving
serving_default_concrete = serving_default.get_concrete_function()

# Save the model with the custom signature
tf.saved_model.save(detection_model, "saved_model", signatures={'serving_default': serving_default_concrete})
