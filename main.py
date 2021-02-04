from datasets.alice_dataset.dataloader import Dataloader
from datasets.aishell_dataset.databuilder.main import AishellDatasetBuilder
import tensorflow as tf
import time, os , datetime
from model_factory import ModelFactory
import pdb
from functools import wraps
from util.mem_check_util import mem_check
import logging
from metrics.wer import calc_wer

  
train_step_signature = [
    tf.TensorSpec(shape=(None,None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class SpeechTranslationTask():
  def __init__(self,args):
    self.args = args

    #--- init dataset ---
    self.asr_ds, self.trans_tokenizer, \
    self.vocab_src_size, self.vocab_tar_size = SpeechTranslationTask.ds_builder(args)
    self.trans_tokenizer.index_word.update({0:"<pad>"})

    #--- init model ---
    self.model, \
    self.optimizer, self.loss_object = SpeechTranslationTask.model_builder(args, self.vocab_src_size, self.vocab_tar_size)
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

    #--- init checkpoint manager  ---
    self.checkpoint = tf.train.Checkpoint(
      optimizer=self.optimizer,
      model=self.model
    )

    self.checkpoint_name = self.args.checkpoint_name + self.args.corpus
    self.checkpoint_dir = self.args.checkpoint_dir
    self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir,checkpoint_name=self.checkpoint_name, max_to_keep=3)

    #-- init tensorboard ---
    self.tensorboard_dir = args.tensorboard_dir 

    self.summar_dir = os.path.join(self.tensorboard_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + self.checkpoint_name

    if self.args.mode == "train":
      self.batch_sz = self.args.batch_sz
      self.summary_writer = tf.summary.create_file_writer(self.summar_dir)     # 参数为记录文件所保存的目录
      print(f"tensorboard dir:{self.summar_dir}")
    elif self.args.mode == "eval":
      self.batch_sz = 1

  def train(self):
    self.model_load()

    EPOCHS = self.args.epoch
    BATCH_SIZE = self.batch_sz

    for epoch in range(EPOCHS):

      self.train_loss.reset_states()
      self.train_accuracy.reset_states()

      for (batch, (inp, targ)) in enumerate(self.asr_ds.padded_batch(BATCH_SIZE, padded_shapes=([None,None],[None]), drop_remainder=True)):
        step_info = self.train_step(inp, targ)    
        step_output = step_info if step_info else {} # in case of None value
        yield { "_epoch":epoch, "_batch":batch, **step_output}

      # 每 2 个周期（epoch），保存（检查点）一次模型
      if (epoch + 1) % 2 == 0:
        self.model_save()
  
  def eval(self):
    self.model_load(expect_partial=True)
    for inp, targ in self.asr_ds.batch(1).take(35):
      if self.args.eval_teacher:
        yield self.eval_teacher_step(inp,targ)
      else:
        yield self.eval_step(inp,targ)

  def __call__(self):
    if self.args.mode == "train":
      return self.train()
    elif self.args.mode == "eval":
      return self.eval()
    
  @staticmethod
  def ds_builder(args):
    if args.corpus == "alice":
      dl = Dataloader(filename=[args.dataset])
      _, trans_tokenizer = dl.get_alice_text_dataset()
      asr_ds, _ = dl.get_alice_asr_dataset()
    elif args.corpus == "aishell":
      aishell = AishellDatasetBuilder("train", trans_mode="tensor", audio_mode="spec", ds_model="tfrecord")
      trans_tokenizer = aishell.trans_tokenizer
      asr_ds = aishell()
    else:
      raise Exception("unsupported dataset")

    if args.is_audio:
      trans_tokenizer = trans_tokenizer
      vocab_src_size = len(trans_tokenizer.word_index) + 1 
      vocab_tar_size = len(trans_tokenizer.word_index) + 1
    else:
      pass

    return asr_ds, trans_tokenizer, vocab_src_size, vocab_tar_size
  
  @staticmethod
  def model_builder(args, vocab_src_size, vocab_tar_size):
    model = ModelFactory()(args.specdim, vocab_tar_size, args)

    learning_rate = CustomSchedule(256)

    if args.lr_schedule:
      print("learning rate decay")
      optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
      optimizer = tf.keras.optimizers.Adam()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    return model, optimizer, loss_object

  def model_load(self, expect_partial=False):
    latest_ckpt =  tf.train.latest_checkpoint(self.checkpoint_dir)
    if latest_ckpt:
      print("Found existed ckpt, loadding...")
      if expect_partial:
        self.checkpoint.restore(latest_ckpt).expect_partial()
      else:
        self.checkpoint.restore(latest_ckpt)
    else:
      print("Training model from scratch!")

  def model_save(self):
    if self.args.mode == "train":
      self.ckpt_manager.save()
    else:
      print("Not in training mode, abort model save!")

  def loss_function(self, real, pred):
    # mask padding token
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = self.loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)

    # mask the lossed of padding token
    loss_ *= mask

    return tf.reduce_mean(loss_)


  def eval_teacher_step(self, inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    predictions, _ = self.model(inp, tar_inp, 
                                training=True)
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    return  {"_result":" ".join([self.trans_tokenizer.index_word[i.numpy()] for i in predicted_id[0] if i not in (0,1,2)]), "_targ":" ".join([self.trans_tokenizer.index_word[i.numpy()] for i in tar[0] if i not in (0,1,2)])}

  def eval_step(self,inp, targ):
    # 因为目标是英语，输入 transformer 的第一个词应该是
    # 英语的开始标记。
    decoder_input = [1]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(targ.shape[-1]):
      
      

      # predictions.shape == (batch_size, seq_len, vocab_size)
      predictions, attention_weights = self.model(inp, 
                                                  output,
                                                  training=False)

      # 从 seq_len 维度选择最后一个词
      predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
      
      # 如果 predicted_id 等于结束标记，就返回结果
      if predicted_id == 2:
        a = tf.squeeze(output, axis=0), attention_weights
        break
      
      # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
      output = tf.concat([output, predicted_id], axis=-1)

    return  {"_result":" ".join([self.trans_tokenizer.index_word[i.numpy()] for i in output[0] if i not in (0,1,2)]), "_targ":" ".join([self.trans_tokenizer.index_word[i.numpy()] for i in targ[0] if i not in (0,1,2)])}


  # @tf.function(input_signature=train_step_signature)
  def train_step(self, inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
      predictions, _ = self.model(inp, tar_inp, 
                                training=True)
      loss = self.loss_function(tar_real, predictions)
    
    variables = self.model.trainable_variables

    gradients = tape.gradient(loss, variables)    
    self.optimizer.apply_gradients(zip(gradients,variables ))
    
    self.train_loss(loss)
    self.train_accuracy(tar_real, predictions)

    return { "batchLoss": self.train_loss.result(), "trainAcc":self.train_accuracy.result(),"learningRate":self.optimizer._decayed_lr(tf.float32)}

def main():
  from base_option import parser
  args = parser.parse_args([
    "--is_audio", 
    "--dataset","/home/tony/D/corpus/Alicecorpus/alice_asr.tfrecord",
    "--batch_sz","64",
    "--epoch","500",
    "--checkpoint_dir","/home/tony/D/exp/training_checkpoints/",
    "--checkpoint_name","transformer-small-convs",
    "--tensorboard_dir","/home/tony/D/exp/tensorboard",
    "--lr_schedule",
    "--eval_teacher",
    "--corpus","alice",
    "--arch","transformer",
    "--mode","train",
    "--specdim","257"
  ])
  print(args)


  task = SpeechTranslationTask(args)
 
  for idx, info in enumerate(task()):
    
    # --- print info ---
    handle_tensor = lambda value: value.numpy() if isinstance(value, tf.Tensor) else value
    handle_start_udl = lambda key: key[1:] if key.startswith("_") else key
    string_infos = [f"{handle_start_udl(key)}: {handle_tensor(value)}" for key, value in info.items()]
    print(*string_infos, sep=" | ")  
    print("----")
    
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