import psutil

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
