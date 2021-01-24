from .datasets.alice_dataset.dataloader import Dataloader
from .datasets.aishell_dataset.databuilder.main import AishellDatasetBuilder
import tensorflow as tf
import time, os 
from .models import *
import pdb
from functools import wraps
from .util.mem_check_util import mem_check
import logging

class SpeechTranslationTask():
  def __init__(self,args):
    self.args = args

    #-- init tensorboard ---
    self.tensorboard_dir = './tensorboard'
    self.summary_writer = tf.summary.create_file_writer(self.tensorboard_dir)     # 参数为记录文件所保存的目录

    #--- init dataset ---
    self.asr_ds, self.trans_tokenizer, \
    self.vocab_src_size, self.vocab_tar_size = SpeechTranslationTask.ds_builder(args)
   
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

  def __call__(self, EPOCHS, BATCH_SIZE):
    self.model_load()

    for epoch in range(EPOCHS):
      for (batch, (inp, targ)) in enumerate(self.asr_ds.shuffle(BATCH_SIZE*2).padded_batch(BATCH_SIZE, padded_shapes=([None,None],[None]), drop_remainder=True)):
        batch_loss = self.train_step(inp, targ).numpy()    
        yield epoch, batch, batch_loss

      # 每 2 个周期（epoch），保存（检查点）一次模型
      if (epoch + 1) % 2 == 0:
        self.model_save()
    
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

    optimizer = tf.keras.optimizers.Adam()
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
    self.ckpt_manager.save()

  def encoder_init_state(self):
    return self.encoder.initialize_hidden_state(self.args.batch_sz)

  def decoder_init_input(self):
    return [self.trans_tokenizer.word_index['<start>']] * self.args.batch_sz

  def loss_function(self, real, pred):
    # mask padding token
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = self.loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)

    # mask the lossed of padding token
    loss_ *= mask

    return tf.reduce_mean(loss_)

  def train_step(self, inp, targ):
    loss = 0
    
    enc_hidden = self.encoder_init_state()

    with tf.GradientTape() as tape:
      
      with mem_check("执行编码"):
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)

      with mem_check("拷贝编码状态"):
        dec_hidden = enc_hidden

      with mem_check("准备解码输入"):
        dec_input = tf.expand_dims(self.decoder_init_input() , 1)

      # 教师强制 - 将目标词作为下一个输入
      for t in range(1, targ.shape[1]):
        with mem_check("单个时间步解码预测"):
          # 将编码器输出 （enc_output） 传送至解码器
          predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
        with mem_check("单个时间步计算损失"):
          # print(t, targ[:,t].shape, predictions.shape)
          loss += self.loss_function(targ[:, t], predictions)

        with mem_check("准备下个时间步解码输入"):
          # 使用教师强制
          dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = self.encoder.trainable_variables + self.decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    self.optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

  def eval_step(self):
    pass


def main():
  from .base_option import parser
  args = parser.parse_args([
    "--is_audio", 
    "--dataset","/home/tony/D/corpus/Alicecorpus/alice_asr.tfrecord",
    "--batch_sz","256",
    "--epoch","100",
    "--checkpoint_dir","./training_checkpoints"
  ])
  print(args)


  task = SpeechTranslationTask(args)
 
  for idx, (epoch,batch,batch_loss) in enumerate(task(EPOCHS=args.epoch, BATCH_SIZE=args.batch_sz)):

    steps_per_epoch = 324//args.batch_sz

    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss))  

    with task.summary_writer.as_default():
      tf.summary.scalar("batchLoss", batch_loss, step=idx)


if __name__ == "__main__":
  main()