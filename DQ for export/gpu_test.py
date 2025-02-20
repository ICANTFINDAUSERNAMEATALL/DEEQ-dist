import tensorflow as tf

print(tf.__version__)

print(tf.config.list_physical_devices('GPU'))

# Alternatively, for more details:
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid allocation issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT available")

    # You can also check which device is assigned to operations:
    with tf.device('/gpu:0'):  # Try placing a simple operation on the first GPU
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)  # Matrix multiplication
    print(c)
else:
    print("No GPUs found.")