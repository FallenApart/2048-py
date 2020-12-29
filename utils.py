import io
import matplotlib.pyplot as plt
import tensorflow as tf


def fig_to_ftimage(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    plt.close()
    return image
