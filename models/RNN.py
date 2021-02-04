import tensorflow as tf

class PreEncoder(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(PreEncoder, self).__init__()
        self.cnn1 = tf.keras.layers.Conv2D(16,(3,3),strides=(2,2))
        self.cnn2 = tf.keras.layers.Conv2D(16,(3,3),strides=(2,2))
        self.cnns = [ tf.keras.layers.Conv2D(16,(3,3)) for i in range(2)]

        self.output_fc = tf.keras.layers.Dense(output_dim)

    def call(self,x):
        x = tf.expand_dims(x,3)

        origin_shape = x.shape
        x = self.cnn1(x)
        x = self.cnn2(x)
        for conv in self.cnns:
            x = conv(x)
        # x = self.cnn3(x)
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

class Encoder(tf.keras.Model):
  def __init__(self,pre_encoder_input_dim, pre_encoder_output_dim, enc_units, is_audio):
    """如果是音频，前编码是convs，如果是文本，前编码是embedding
    """
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.is_audio = is_audio
    if self.is_audio:
      self.pre_encoder = PreEncoder(pre_encoder_input_dim,pre_encoder_output_dim)
    else:
      self.pre_encoder = tf.keras.layers.Embedding(pre_encoder_input_dim, pre_encoder_output_dim)


    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.pre_encoder(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self, batch_sz):
    return tf.zeros((batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # 用于注意力
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
    x = self.embedding(x)

    # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # 将合并后的向量传送到 GRU
    output, state = self.gru(x)

    # 输出的形状 == （批大小 * 1，隐藏层大小）
    output = tf.reshape(output, (-1, output.shape[2]))

    # 输出的形状 == （批大小，vocab）
    x = self.fc(output)

    return x, state, attention_weights


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # 隐藏层的形状 == （批大小，隐藏层大小）
    # hidden_with_time_axis 的形状 == （批大小，1，隐藏层大小）
    # 这样做是为了执行加法以计算分数  
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # 分数的形状 == （批大小，最大长度，1）
    # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V
    # 在应用 self.V 之前，张量的形状是（批大小，最大长度，单位）
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # 注意力权重 （attention_weights） 的形状 == （批大小，最大长度，1）
    attention_weights = tf.nn.softmax(score, axis=1)

    # 上下文向量 （context_vector） 求和之后的形状 == （批大小，隐藏层大小）
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
