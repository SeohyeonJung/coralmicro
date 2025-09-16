# Multi-DNN FreeRTOS 프로젝트용 CMakeLists.txt
# coralmicro/apps/multi_dnn/ 디렉토리에 위치

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(multi_dnn_SOURCES 
    main.cc
    # person_detect.cc
    # mcu.cc
)

set(multi_dnn_LINK_LIBRARIES
    libs_base-m7_freertos
    libs_jpeg_m7
    libs_tensorflow-m7
)

add_executable_m7(multi_dnn
    ${multi_dnn_SOURCES}
    DATA
    ${PROJECT_SOURCE_DIR}/models/bert_tiny_tanh_manual_INT8_bs1_L128_outint8.tflite
    # ${PROJECT_SOURCE_DIR}/models/person.raw
    ${PROJECT_SOURCE_DIR}/models/attention_mask.raw
    ${PROJECT_SOURCE_DIR}/models/input_ids.raw
    ${PROJECT_SOURCE_DIR}/models/token_type_ids.raw
)

target_link_libraries(multi_dnn
    ${multi_dnn_LINK_LIBRARIES}
)


# TPU Edge TPU/M7) build
set(multi_dnn_tpu_SOURCES
    main_TPU.cc
)

set(multi_dnn_tpu_LINK_LIBRARIES
    libs_base-m7_freertos
    libs_tensorflow-m7
    # libs_jpeg_m7
    libs_tpu_freertos
    libs_tpu_task_freertos
)

add_executable_m7(multi_dnn_tpu
    ${multi_dnn_tpu_SOURCES}
    DATA
    # ${PROJECT_SOURCE_DIR}/models/emo_mobilebert_int8_edgetpu.tflite
    ${PROJECT_SOURCE_DIR}/models/bert_tiny_relu_fp32_bs1_L128_edgetpu.tflite
    # ${PROJECT_SOURCE_DIR}/models/apple.raw
    # ${PROJECT_SOURCE_DIR}/models/banana_int8_224.raw
    # ${PROJECT_SOURCE_DIR}/models/mobilebert_qa_input.raw
    ${PROJECT_SOURCE_DIR}/models/input_ids.raw
    ${PROJECT_SOURCE_DIR}/models/attention_mask.raw
    ${PROJECT_SOURCE_DIR}/models/token_type_ids.raw
)

target_link_libraries(multi_dnn_tpu
    ${multi_dnn_tpu_LINK_LIBRARIES}
)