from dataloader import Dataloader
import tensorflow as tf
import time, os 
from models import *
import pdb
import psutil
from functools import wraps

LOG = False
def _print(*args,**kwargs):
  if LOG:
    print(*args,**kwargs)

class mem_check:
  def __init__(self, comment="", do_echo_used=False):
    self.do_echo_used = do_echo_used
    if comment:
      _print(comment)
  
  def __enter__(self):
    self.m0 = psutil.virtual_memory().used
    if self.do_echo_used:
      _print(f"等待语句执行完成后，检查内存增量, m0:{self.humanize(self.m0)}")


  def __exit__(self, exc_type, exc_value, exc_tb):
    self.m1 = psutil.virtual_memory().used
    mem_increased = self.m1 - self.m0
    if self.do_echo_used:
      _print(f"m1:{self.humanize(self.m1)}")
    _print(f"内存增量:{self.humanize(mem_increased)}")
  def humanize(self, bytes_cnt):
    return f"{(bytes_cnt)/1024**2}MB"



def check_mem_increase(f):
  @wraps(f)
  def decorated(*args,**kwargs):
    _print(f"准备运行函数:{f.__name__}")
    with mem_check():
      result = f(*args,**kwargs)
    return result
  return decorated

@tf.function(experimental_relax_shapes=True)
def train_step(inp, targ, enc_hidden, is_audio=False):
  loss = 0
  with tf.GradientTape() as tape:
    
    with mem_check("执行编码"):
      enc_output, enc_hidden = encoder(inp, enc_hidden)

    with mem_check("拷贝编码状态"):
      dec_hidden = enc_hidden

    with mem_check("准备解码输入"):
      dec_input = tf.expand_dims([trans_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    # 教师强制 - 将目标词作为下一个输入
    for t in range(1, targ.shape[1]):
      with mem_check("单个时间步解码预测"):
        # 将编码器输出 （enc_output） 传送至解码器
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      with mem_check("单个时间步计算损失"):
        # print(t, targ[:,t].shape, predictions.shape)
        loss += loss_function(targ[:, t], predictions)

      with mem_check("准备下个时间步解码输入"):
        # 使用教师强制
        dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

def main():
  from base_option import parser
  args = parser.parse_args()
  IS_AUDIO = args.isaudio
  EPOCHS = 10
  dl = Dataloader(filename=[args.dataset])
  alice_text_ds, trans_tokenizer = dl.get_alice_text_dataset()
  # alice_asr_ds,_ = dl.get_alice_asr_dataset()

  BUFFER_SIZE = trans_tokenizer.document_count
  embedding_dim = 128
  units = 512
  vocab_src_size = len(trans_tokenizer.word_index) + 1 
  vocab_tar_size = len(trans_tokenizer.word_index) + 1
  pre_encoder_output_dim = 128

  if IS_AUDIO:
    pre_encoder_input_dim = 257
    BATCH_SIZE = 1

  else:
    pre_encoder_input_dim = vocab_src_size
    BATCH_SIZE = 32
    EPOCHS = 100

  steps_per_epoch = 324//BATCH_SIZE


  encoder = Encoder(pre_encoder_input_dim=pre_encoder_input_dim, pre_encoder_output_dim=pre_encoder_output_dim,enc_units=units, batch_sz=BATCH_SIZE, is_audio=IS_AUDIO)
  decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

  optimizer = tf.keras.optimizers.Adam()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')


  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                  encoder=encoder,
                                  decoder=decoder)
  for epoch in range(EPOCHS):

    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    # for (batch, (inp, targ)) in enumerate(alice_asr_ds.shuffle(BATCH_SIZE*2).padded_batch(BATCH_SIZE, padded_shapes=([None,None],[None]), drop_remainder=True)):
    for (batch,targ) in enumerate(alice_text_ds.shuffle(BATCH_SIZE*2).padded_batch(BATCH_SIZE, padded_shapes=[None], drop_remainder=True)):
      # pdb.set_trace()
      if IS_AUDIO:
        batch_loss = train_step(inp, targ, enc_hidden)
      else:
        batch_loss = train_step(targ, targ, enc_hidden)
        
      total_loss += batch_loss.numpy()
      if batch % 1 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                      batch,
                                                      batch_loss.numpy()))
    # 每 2 个周期（epoch），保存（检查点）一次模型
    if (epoch + 1) % 2 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))




if __name__ == "__main__":
  main()