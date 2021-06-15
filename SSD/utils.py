import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Reshape
from Default_box_layers import DefBoxes

num_class = 21
variance = [0.1, 0.1, 0.2, 0.2]
aspect_ratio_4 = [1.0, 2.0, 0.5]
aspect_ratio_6 = [1.0, 2.0, 0.5, 3.0, 0.33]

def get_pred_4(input, scale1, next_scale1):
  
    conf_4 = Conv2D(4 * num_class, (3,3), padding='same')(input)
    loc_4 = Conv2D(4 * 4, (3,3), padding='same')(input)
    def_box_4 = DefBoxes((300, 300, 3), scale1, next_scale1, aspect_ratio_4, variance)(input)

    conf_4 = Reshape((-1, num_class))(conf_4)
    loc_4 = Reshape((-1, 4))(loc_4)
    def_box_4 = Reshape((-1, 8))(def_box_4)

    return conf_4, loc_4, def_box_4

def get_pred_6(input, scale1, next_scale1):
  
    conf_6 = Conv2D(6 * num_class, (3,3), padding='same')(input)
    loc_6 = Conv2D(6 * 4, (3,3), padding='same')(input)
    def_box_6 = DefBoxes((300, 300, 3), scale1, next_scale1, aspect_ratio_6, variance)(input)

    conf_6 = Reshape((-1, num_class))(conf_6)
    loc_6 = Reshape((-1, 4))(loc_6)
    def_box_6 = Reshape((-1, 8))(def_box_6)

    return conf_6, loc_6, def_box_6