""" single learner multiple actors """
import os

import tensorflow as tf
from six.moves import range


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer('num_actors', 2, 'Number of actors.')
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'],
                  'Job name. Ignored when task is set to -1.')

dtype = tf.float32
shape = [3, 2]
batch_size = 6
max_iter = 1000*1000
log_dir = 'log_slma/'
shared_job_device = '/job:learner/task:0'
local_job_device = '/job:{}/task:{}'.format(FLAGS.job_name, FLAGS.task)
is_actor_fn = lambda i: FLAGS.job_name == 'actor' and i == FLAGS.task
is_learner = FLAGS.job_name == 'learner'
cluster = tf.train.ClusterSpec({
  'actor': ['localhost:%d' % (8001 + i) for i in range(FLAGS.num_actors)],
  'learner': ['localhost:8000']
})


def build_net(x_input):
  y = tf.layers.dense(x_input, units=1)
  return y


def main(_):
  # create the graph
  with tf.device(local_job_device):
    with tf.device(shared_job_device):
      queue = tf.FIFOQueue(capacity=1, dtypes=[dtype], shapes=[shape],
                          shared_name='replay_buffer')

    enq_ops = []
    for i in range(FLAGS.num_actors):
      if is_actor_fn(i):
        tmp = tf.random_normal(shape=shape, dtype=dtype,
                               name='input{}'.format(i))
        input = tf.train.input_producer(tmp).dequeue_many(shape[0]) # slight hacking
        with tf.device(shared_job_device):
          enq_ops.append(queue.enqueue(input, name='enq_input_{}'.format(i)))

    # build the learner
    if is_learner:
      # Create global step.
      g_step = tf.get_variable(
        'global_step',
        initializer=tf.zeros_initializer(),
        shape=[],
        dtype=tf.int64,
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

      dequeued = queue.dequeue_many(batch_size, name='dequeued')

      y = build_net(dequeued)
      loss = tf.reduce_mean(y)
      train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(
        loss=loss,
        global_step=g_step
      )

  # run the session
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task)
  config = tf.ConfigProto(allow_soft_placement=True,
                          device_filters=[shared_job_device, local_job_device])
  with tf.train.MonitoredTrainingSession(
          server.target,
          is_chief=is_learner,
          checkpoint_dir=os.path.join(log_dir, 'z'),
          config=config) as sess:
    if is_learner:
      summary_writer = tf.summary.FileWriterCache.get(os.path.join(
        log_dir, 'z'))
      summary = tf.summary.Summary()

      for j in range(max_iter):
        print('iter {}'.format(j))
        out_loss, _ = sess.run([loss, train_op])
        print(out_loss)

        summary.value.add(tag='learner/xyz', simple_value=out_loss)
        summary_writer.add_summary(summary, j)
    else:
      summary_writer = tf.summary.FileWriterCache.get(os.path.join(
        log_dir, 'z_{}'.format(FLAGS.task)))
      summary = tf.summary.Summary()
      local_step = 0
      while True:
        fetched = sess.run([input, enq_ops])
        print('actor {}, step {}: enqueue'.format(FLAGS.task, local_step))

        summary.value.add(tag='actor/xyz', simple_value=fetched[0].mean())
        summary_writer.add_summary(summary, local_step)

        local_step += 1


if __name__ == '__main__':
  tf.app.run()
