import tensorflow as tf

dtype = tf.float32
shape = [3]
n_thread = 8
batch_size = 4
max_iter = 1000
log_dir = 'log_my_q/'

queue = tf.FIFOQueue(capacity=1, dtypes=[dtype], shapes=[shape],
                     shared_name='replay_buffer')

enq_ops = []
for i in range(n_thread):
  input = tf.random_normal(shape=shape, dtype=dtype, name='input{}'.format(i))
  enq_ops.append(queue.enqueue(input, name='enq_input_{}'.format(i)))

tf.train.add_queue_runner(tf.train.QueueRunner(queue, enq_ops))

dequeued = queue.dequeue_many(batch_size, name='dequeued')
#dequeued = queue.dequeue(name='dequeued')

with tf.train.MonitoredTrainingSession() as sess:
  summary_writer = tf.summary.FileWriterCache.get(log_dir)
  for j in range(max_iter):
    print('iter {}'.format(j))
    outputs = sess.run(dequeued)
    print(outputs.shape)

    summary = tf.summary.Summary()
    summary.value.add(tag='xyz', simple_value=outputs.mean())
    summary_writer.add_summary(summary, j)
