import tensorflow as tf
class PreEncoder(tf.keras.Model):
    def __init__(self, input_dim, output_dim, batch_sz):
        super(PreEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.cnn1 = tf.keras.layers.Conv2D(4,(3,3),strides=(2,2))
        self.cnn2 = tf.keras.layers.Conv2D(8,(3,3),strides=(2,2))
        self.cnn3 = tf.keras.layers.Conv2D(16,(3,3),strides=(2,2))

        self.output_fc = tf.keras.layers.Dense(output_dim)

    def call(self,x):
        x = tf.expand_dims(x,3)

        origin_shape = x.shape
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        # output: x.shape : (bsz, spec_dim, time, filter)
        x = tf.transpose(x,perm=[0,2,1,3])
        # output: x.shape : (bsz, time , spec_dim, filter)
        try:
            x = tf.reshape(x,[x.shape[0],x.shape[1],-1])
        except TypeError as e:
            print(origin_shape)
            print(x.shape)
            raise e
        # output: x.shape : (bsz, time , spec_dim * filter)
        x = self.output_fc(x)
        return x