from datasets.alice_dataset.dataloader import Dataloader
from datasets.aishell_dataset.databuilder.main import AishellDatasetBuilder
import tensorflow as tf
import time, os , datetime
from models import *
import pdb
from functools import wraps
from util.mem_check_util import mem_check
import logging
from metrics.wer import calc_wer

  

class SpeechTranslationTask():
  def __init__(self,args):
    self.args = args

    #-- init tensorboard ---
    self.tensorboard_dir = args.tensorboard_dir 

    summar_dir = os.path.join(self.tensorboard_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"tensorboard dir:{summar_dir}")
    self.summary_writer = tf.summary.create_file_writer(summar_dir)     # 参数为记录文件所保存的目录

    #--- init dataset ---
    self.asr_ds, self.trans_tokenizer, \
    self.vocab_src_size, self.vocab_tar_size = SpeechTranslationTask.ds_builder(args)
    self.trans_tokenizer.index_word.update({0:"<pad>"})

    #--- init model ---
    self.encoder, self.decoder, \
    self.optimizer, self.loss_object = SpeechTranslationTask.model_builder(args, self.vocab_src_size, self.vocab_tar_size)
    
    #--- init checkpoint manager  ---
    self.checkpoint = tf.train.Checkpoint(
      optimizer=self.optimizer,
      encoder=self.encoder,
      decoder=self.decoder
    )
    self.checkpoint_prefix = os.path.join(self.args.checkpoint_dir, "ckpt")
    self.checkpoint_dir = self.args.checkpoint_dir
    self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir, max_to_keep=3)

    if self.args.mode == "train":
      self.batch_sz = self.args.batch_sz
    elif self.args.mode == "eval":
      self.batch_sz = 1

  def train(self):
    self.model_load()

    EPOCHS = self.args.epoch
    BATCH_SIZE = self.batch_sz

    for epoch in range(EPOCHS):
      for (batch, (inp, targ)) in enumerate(self.asr_ds.shuffle(BATCH_SIZE*5).padded_batch(BATCH_SIZE, padded_shapes=([None,None],[None]), drop_remainder=True)):
        step_info = self.train_step(inp, targ)    
        step_output = step_info if step_info else {} # in case of None value
        yield { "_epoch":epoch, "_batch":batch, **step_output}

      # 每 2 个周期（epoch），保存（检查点）一次模型
      if (epoch + 1) % 2 == 0:
        self.model_save()
  
  def eval(self):
    for inp, targ in self.asr_ds.batch(1).take(5):
      yield self.eval_step(inp,targ)

  def __call__(self):
    if self.args.mode == "train":
      return self.train()
    elif self.args.mode == "eval":
      return self.eval()
    
  @staticmethod
  def ds_builder(args):
    # dl = Dataloader(filename=[args.dataset])
    # alice_text_ds, trans_tokenizer = dl.get_alice_text_dataset()
    # alice_asr_ds,_ = dl.get_alice_asr_dataset()

    aishell = AishellDatasetBuilder("train", trans_mode="tensor", audio_mode="spec", ds_model="tfrecord")
    trans_tokenizer = aishell.trans_tokenizer
    asr_ds = aishell()
    if args.is_audio:
      trans_tokenizer = trans_tokenizer
      vocab_src_size = len(trans_tokenizer.word_index) + 1 
      vocab_tar_size = len(trans_tokenizer.word_index) + 1
    else:
      pass

    return asr_ds, trans_tokenizer, vocab_src_size, vocab_tar_size
  
  @staticmethod
  def model_builder(args, vocab_src_size, vocab_tar_size):
    dec_embedding_dim = 128
    encoder_units = 256
    decoder_units = 256
    pre_encoder_output_dim = 128

    if args.is_audio:
      pre_encoder_input_dim = 40
    else:
      pre_encoder_input_dim = args.vocab_src_size

    encoder = Encoder(
      pre_encoder_input_dim=pre_encoder_input_dim,
      pre_encoder_output_dim=pre_encoder_output_dim,
      enc_units=encoder_units,
      is_audio=args.is_audio
    )

    decoder = Decoder(
      vocab_size=vocab_tar_size, 
      embedding_dim=dec_embedding_dim, 
      dec_units=decoder_units 
    )
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.005, decay_steps=20000, decay_rate=.90)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    return encoder, decoder, optimizer, loss_object

  def model_load(self):
    latest_ckpt =  tf.train.latest_checkpoint(self.checkpoint_dir)
    if latest_ckpt:
      print("Found existed ckpt, loadding...")
      self.checkpoint.restore(latest_ckpt)
    else:
      print("Training model from scratch!")

  def model_save(self):
    if self.args.mode == "train":
      self.ckpt_manager.save()
    else:
      print("Not in training mode, abort model save!")

  def encoder_init_state(self):
    return self.encoder.initialize_hidden_state(self.batch_sz)

  def decoder_init_input(self):
    return [self.trans_tokenizer.word_index['<start>']] * self.batch_sz

  def loss_function(self, real, pred):
    # mask padding token
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = self.loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)

    # mask the lossed of padding token
    loss_ *= mask

    return tf.reduce_mean(loss_)


  def eval_step(self,inp, targ):
    enc_hidden = self.encoder_init_state()

    # 执行编码
    enc_output, enc_hidden = self.encoder(inp, enc_hidden)
    # 拷贝编码状态
    dec_hidden = enc_hidden
    # 准备解码输入
    dec_input = tf.expand_dims(self.decoder_init_input() , 1)

    result =  ''      

    # 教师强制 - 将目标词作为下一个输入
    for t in range(1, targ.shape[1]):
      # 单个时间步解码预测
      # 将编码器输出 （enc_output） 传送至解码器
      predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_output)
      predicted_id = tf.argmax(predictions[0]).numpy()

      # 准备下个时间步解码输入
      # 预测的 ID 被输送回模型
      # dec_input = tf.expand_dims([predicted_id], 0)
      # 使用教师强制
      dec_input = tf.expand_dims(targ[:, t], 1)

      result += self.trans_tokenizer.index_word[predicted_id]

      if targ[:,t] == 0:
        break
    
    return  {"_result":result, "_targ":"".join([self.trans_tokenizer.index_word[i.numpy()] for i in targ[0] if i not in (0,1,2)])}



  def train_step(self, inp, targ):
    loss = 0
    
    enc_hidden = self.encoder_init_state()

    with tf.GradientTape() as tape:
      
      # 执行编码
      enc_output, enc_hidden = self.encoder(inp, enc_hidden)
      # 拷贝编码状态
      dec_hidden = enc_hidden
      # 准备解码输入
      dec_input = tf.expand_dims(self.decoder_init_input() , 1)
      # 教师强制 - 将目标词作为下一个输入
      for t in range(1, targ.shape[1]):
        # 单个时间步解码预测
        # 将编码器输出 （enc_output） 传送至解码器
        predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
        # 单个时间步计算损失
        loss += self.loss_function(targ[:, t], predictions)
        # 准备下个时间步解码输入
        # 使用教师强制
        dec_input = tf.expand_dims(targ[:, t], 1)
        a = tf.reduce_sum(targ[:,t])
        if a==0:
          break

    batch_loss = (loss / int(targ.shape[1]))

    variables = self.encoder.trainable_variables + self.decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    self.optimizer.apply_gradients(zip(gradients, variables))
    return { "batchLoss": batch_loss, "learningRate":self.optimizer._decayed_lr(tf.float32)}

def main():
  from base_option import parser
  args = parser.parse_args([
    "--is_audio", 
    "--dataset","/home/tony/D/corpus/Alicecorpus/alice_asr.tfrecord",
    "--batch_sz","256",
    "--epoch","100",
    "--checkpoint_dir","/home/tony/D/exp/training_checkpoints",
    "--tensorboard_dir","/home/tony/D/exp/tensorboard",
    "--mode","eval"
  ])
  print(args)


  task = SpeechTranslationTask(args)
 
  for idx, info in enumerate(task()):
    
    # --- print info ---
    handle_tensor = lambda value: value.numpy() if isinstance(value, tf.Tensor) else value
    handle_start_udl = lambda key: key[1:] if key.startswith("_") else key
    string_infos = [f"{handle_start_udl(key)}: {handle_tensor(value)}" for key, value in info.items()]
    print(*string_infos, sep=" | ")  
    
    # --- write tensorboard ---
    if args.mode == "train":
      with task.summary_writer.as_default():
        for key in info:
          if not key.startswith("_"):
            tf.summary.scalar(key, info[key], step=idx)

  # task = SpeechTranslationTask(args)
  # for spec, trans in task.asr_ds.take(5):
  #   indexs = list(trans.numpy())
  #   task.trans_tokenizer.index_word.update({0:"<pad>"})
  #   a = "".join([task.trans_tokenizer.index_word[i] for i in indexs if i != 0])
  #   print(a)

if __name__ == "__main__":
  main()