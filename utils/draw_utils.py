import tensorflow as tf
from PIL import ImageDraw


def draw_rpn_output(image, roi_bboxes, roi_scores, top_n):
    image = tf.squeeze(image, axis=0)
    image = tf.keras.preprocessing.image.array_to_img(image)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    y1 = roi_bboxes[0][..., 0] * height
    x1 = roi_bboxes[0][..., 1] * width
    y2 = roi_bboxes[0][..., 2] * height
    x2 = roi_bboxes[0][..., 3] * width

    denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))
    _, top_indices = tf.nn.top_k(roi_scores[0], top_n)
    selected_rpn_bboxes = tf.gather(denormalized_box, top_indices, batch_dims=0)

    for bbox in selected_rpn_bboxes:
        y1, x1, y2, x2 = tf.split(bbox, 4, axis=-1)
        draw.rectangle((x1, y1, x2, y2), outline=234, width=3)

    return image


def draw_dtn_output(
    image,
    final_bboxes,
    labels,
    final_labels,
    final_scores,
    colors,
):
    image = tf.squeeze(image, axis=0)
    final_bboxes = tf.squeeze(final_bboxes, axis=0)
    final_labels = tf.squeeze(final_labels, axis=0)
    final_scores = tf.squeeze(final_scores, axis=0)

    image = tf.keras.preprocessing.image.array_to_img(image)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    y1 = final_bboxes[..., 0] * height
    x1 = final_bboxes[..., 1] * width
    y2 = final_bboxes[..., 2] * height
    x2 = final_bboxes[..., 3] * width

    denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

    for index, bbox in enumerate(denormalized_box):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis=-1)
        width = x2 - x1
        height = y2 - y1

        label_index = int(final_labels[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    return image
