from mcunet.model_zoo import net_id_list, build_model, download_tflite
print(net_id_list)  # the list of models in the model zoo

# pytorch fp32 model
model, image_size, description = build_model(net_id="mcunet-in3", pretrained=True)  # you can replace net_id with any other option from net_id_list

# download tflite file to tflite_path
tflite_path = download_tflite(net_id="mcunet-in3")
