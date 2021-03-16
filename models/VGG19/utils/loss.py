# Loss functions
import tensorflow as tf

def content_loss(placeholder,content,weight):
    assert placeholder.shape == content.shape
    return weight*tf.reduce_mean(tf.square(placeholder-content))

def gram_matrix(x):
    gram=tf.linalg.einsum('bijc,bijd->bcd', x, x)
    return tf.cast(gram,tf.float32)/tf.cast(x.shape[1]*x.shape[2]*x.shape[3],tf.float32)

def style_loss(placeholder,style,weight):
    assert placeholder.shape == style.shape
    s=gram_matrix(style)
    p=gram_matrix(placeholder)
    return weight*tf.reduce_mean(tf.square(s-p))

def perceptual_loss(predicted_activations,content_activations,style_activations,content_weight,style_weight,content_layers_weights,style_layer_weights):
    # Content loss
    pred_content=predicted_activations["content"]
    c_loss=tf.add_n([content_loss(pred_content[name],content_activations[name],content_layers_weights[i]) for i,name in enumerate(pred_content.keys())])
    c_loss=c_loss*content_weight

    # Style loss
    pred_style=predicted_activations["style"]
    s_loss=tf.add_n([style_loss(pred_style[name],style_activations[name],style_layer_weights[i]) for i,name in enumerate(pred_style.keys())])
    s_loss=s_loss*style_weight

    return c_loss + s_loss, c_loss, s_loss