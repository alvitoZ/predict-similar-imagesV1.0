import tensorflow as tf
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = 'tf2-preview_mobilenet_v2_feature_vector_4/saved_model.pb', 
    input_arrays = ['Input_Tensor_Name'],
    output_arrays = ['Output_Tensor_Name'] 
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)