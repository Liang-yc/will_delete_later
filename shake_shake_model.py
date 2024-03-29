import keras
from keras import backend as K
from keras.layers import Layer


class ShakeShake(Layer):
    """ Shake-Shake-Image Layer """

    def __init__(self, **kwargs):
        super(ShakeShake, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ShakeShake, self).build(input_shape)

    def call(self, x):
        # unpack x1 and x2
        assert isinstance(x, list)
        x1, x2 = x
        # create alpha and beta
        batch_size = K.shape(x1)[0]
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        beta = K.random_uniform((batch_size, 1, 1, 1))
        # shake-shake during training phase
        def x_shake():
            return beta * x1 + (1 - beta) * x2 + K.stop_gradient((alpha - beta) * x1 + (beta - alpha) * x2)
        # even-even during testing phase
        def x_even():
            return 0.5 * x1 + 0.5 * x2
        return K.in_train_phase(x_shake, x_even)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]

l2 = keras.regularizers.l2(1e-4)


def create_residual_branch(x, filters, stride):
    """ Regular Branch of a Residual network: ReLU -> Conv2D -> BN repeated twice """
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2,
                            use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2,
                            use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    return x


def create_residual_shortcut(x, filters, stride):
    """ Shortcut Branch used when downsampling from Shake-Shake regularization """
    x = keras.layers.ReLU()(x)
    x1 = keras.layers.Lambda(lambda y: y[:, 0:-1:stride, 0:-1:stride, :])(x)
    x1 = keras.layers.Conv2D(filters // 2, kernel_size=1, strides=1, padding='valid',
                             kernel_initializer='he_normal', kernel_regularizer=l2,
                             use_bias=False)(x1)
    x2 = keras.layers.Lambda(lambda y: y[:, 1::stride, 1::stride, :])(x)
    x2 = keras.layers.Conv2D(filters // 2, kernel_size=1, strides=1, padding='valid',
                             kernel_initializer='he_normal', kernel_regularizer=l2,
                             use_bias=False)(x2)
    x = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.BatchNormalization()(x)
    return x


def create_residual_block(x, filters, stride=1):
    """ Residual Block with Shake-Shake regularization and shortcut """
    x1 = create_residual_branch(x, filters, stride)
    x2 = create_residual_branch(x, filters, stride)
    if stride > 1: x = create_residual_shortcut(x, filters, stride)
    return keras.layers.Add()([x, ShakeShake()([x1, x2])])


def create_residual_layer(x, filters, blocks, stride):
    """ Layer repeating Residual Blocks """
    x = create_residual_block(x, filters, stride)
    for i in range(1, blocks):
        x = create_residual_block(x, filters, 1)
    return x


def create_shakeshake_cifar(n_classes, n_blocks=[5, 5, 5], activation='softmax',include_top=True,x_in=None):
    """ Residual Network with Shake-Shake regularization modeled after ResNetCifar10 """
    # Input and first convolutional layer
    if x_in==None:
        x_in = keras.layers.Input(shape=(28,28,1))
    x = keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2,
                            use_bias=False)(x_in)
    x = keras.layers.BatchNormalization()(x)
    # Three stages of four residual blocks
    x = create_residual_layer(x, 16, n_blocks[0], 1)
    x = create_residual_layer(x, 32, n_blocks[1], 2)
    x = create_residual_layer(x, 64, n_blocks[2], 2)
    # Output pooling and dense layer
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    if include_top:
        x_out = keras.layers.Dense(n_classes, activation=activation, kernel_initializer='he_normal')(x)
        return keras.models.Model(x_in, x_out)
    return x


def create_shakeshake_imagenet(n_classes, n_blocks=[3, 4, 6, 3], activation='softmax'):
    """ Residual Network with Shake-Shake regularization modeled after ResNet32 """
    # Input and first convolutional layer
    x_in = keras.layers.Input(shape=(224, 224, 3))
    x = keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2,
                            use_bias=False)(x_in)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    # Three stages of four residual blocks
    x = create_residual_layer(x, 64, n_blocks[0], 1)
    x = create_residual_layer(x, 128, n_blocks[1], 2)
    x = create_residual_layer(x, 256, n_blocks[2], 2)
    x = create_residual_layer(x, 512, n_blocks[3], 2)
    # Output pooling and dense layer
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x_out = keras.layers.Dense(n_classes, activation=activation, kernel_initializer='he_normal')(x)
    return keras.models.Model(x_in, x_out)