import tensorflow as tf

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs found:")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("No GPUs found.")

# Check TensorFlow's device placement
from tensorflow.python.framework import ops
ops.reset_default_graph()
print("TensorFlow version:", tf.__version__)
print("Is GPU available:", tf.test.is_gpu_available())
print("List of devices:")
print([device.name for device in tf.config.list_logical_devices()])