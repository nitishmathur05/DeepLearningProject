import tfcoreml as tf_converter
tf_model_path = 'output_graph_oct_16.pb'
input_tensor_shapes = {'image:0':[1,224,224,3]} 
image_labels = ['non_porn','porn']
# Convert graph to mlmodel
coreml_model = tf_converter.convert(
    tf_model_path=tf_model_path,
    mlmodel_path='output.mlmodel',
    input_name_shape_dict=input_tensor_shapes,
    output_feature_names=['final_result:0'],
    image_input_names=['input:0'],
    class_labels=image_labels,
    image_scale = 1.0/255.0
)