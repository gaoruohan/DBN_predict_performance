import tensorflow as tf
import numpy as np

def predict(data):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('saved_model/model.ckpt.meta')
        new_saver.restore(sess, 'saved_model/model.ckpt')
        # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
        y = tf.get_collection('predict')[0]

        graph = tf.get_default_graph()
        # for op in graph.get_operations():
        #     print(op.name, op.values())

        # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
        input_x = graph.get_tensor_by_name("Placeholder:0")
        # keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        input=data
        # 使用y进行预测
        print(sess.run(y, feed_dict={input_x:input }))

if __name__=="__main__":
    data=np.array([[0.17783, 0.00, 9.690, 0, 0.5850, 5.5690, 73.50, 2.3999, 6, 391.0, 19.20, 395.77, 15.10]])

    predict(data)