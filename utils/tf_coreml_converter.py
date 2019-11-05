import tfcoreml
tfcoreml.convert(tf_model_path='<frozen_graph_name>.pb',
        mlmodel_path='<relevant_name>.mlmodel',
        input_name_shape_dict={"input:0":[1,224,224,3]},
        output_feature_names=['final_result:0'],
        image_input_names = ['input:0'],
        class_labels = ['non porn','porn'],
        red_bias = -1,
        green_bias = -1,
        blue_bias = -1,
        image_scale = 2.0/255.0
)