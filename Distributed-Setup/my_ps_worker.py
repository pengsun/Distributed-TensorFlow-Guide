"""Simple example with one parameter server and one worker.

Author: Tommy Mulc
"""


from __future__ import print_function
import tensorflow as tf
import argparse
import time
import os


FLAGS = None
log_dir = 'low_wp/z'

def main():
  # Distributed Baggage
  cluster = tf.train.ClusterSpec({
        'ps':['localhost:2222'],
        'worker':['localhost:2223', 'localhost:2224']
        }) #lets this node know about all other nodes
  if FLAGS.job_name == 'ps': #checks if parameter server
    server = tf.train.Server(cluster,job_name="ps",task_index=FLAGS.task_index)
    server.join()
  else:
    is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
    server = tf.train.Server(cluster,job_name="worker",task_index=FLAGS.task_index)

    # Graph
    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
      a = tf.Variable(tf.truncated_normal(shape=[2]),dtype=tf.float32)
      b = tf.Variable(tf.truncated_normal(shape=[2]),dtype=tf.float32)
      c=a+b

      target = tf.constant(100.,shape=[2],dtype=tf.float32)
      loss = tf.reduce_mean(tf.square(c-target))

      opt = tf.train.GradientDescentOptimizer(.0001).minimize(loss)

    summary_writer = tf.summary.FileWriterCache.get(
      log_dir + '_{}'.format(FLAGS.task_index))

    # Session
    # Supervisor
    sv = tf.train.Supervisor(logdir=os.getcwd()+log_dir,is_chief=is_chief,save_model_secs=30)
    sess = sv.prepare_or_wait_for_session(server.target)
    for i in range(1000):
      if sv.should_stop(): break
      sess.run(opt)

      if i % 10 == 0:
        r = sess.run(c)
        print(r)

        # summary
        summary = tf.summary.Summary()
        summary.value.add(
          tag='xyz', simple_value=r.mean())
        summary_writer.add_summary(summary, i)

        time.sleep(.1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
    	"--job_name",
    	type=str,
    	default="",
    	help="One of 'ps', 'worker'"
    )
  # Flags for defining the tf.train.Server
  parser.add_argument(
    	"--task_index",
    	type=int,
    	default=0,
    	help="Index of task within the job"
    )
  FLAGS, unparsed = parser.parse_known_args()
  main()
