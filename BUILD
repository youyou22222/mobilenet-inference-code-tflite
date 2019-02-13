
# Description:
#TensorFlow TensorFlow Lite Example Label Image.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0
#load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "gen_selected_ops")

exports_files(
    ["version_script.lds",],
    visibility = ["//visibility:public"],
)

LINKER_SCRIPT = "//tensorflow/MobileNetv1_tflite:version_script.lds"
cc_binary(
    name = "libcnnwrapper.so",
    srcs = [
        "mobilenet.cc",
        "mobilenet.h",
        "mobilenet_wrapper.h",
        "mobilenet_wrapper.cc",
        
    ],
    copts = tflite_copts()+[
            "-Os",
            "-ffunction-sections",
            "-fdata-sections",
            #"-flto",
            ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-latomic",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-landroid",
            "-Wl,-soname,libcnnwrapper.so",
            "-llog",
            "-Wl,--gc-sections",
            "-Wl,--version-script",  # This line must be directly followed by LINKER_SCRIPT.
            "$(location {})".format(LINKER_SCRIPT),
        ],
        "//conditions:default": [],
    }),
    deps = [
       "//tensorflow/lite/kernels:builtin_ops",
       LINKER_SCRIPT,
      #"//tensorflow/contrib/lite/tools:mutable_op_resolver",
    ],

    linkshared = 1,
    linkstatic = 1,
)


cc_library(
    name = "libtflite",
    srcs = ["libcnnwrapper.so"],
    visibility = ["//visibility:public"],
)


cc_binary(
    name="mv1_test",
    srcs = ["mobilenet_wrapper.h", "cwrapper_test.cc"],
    deps = [":libtflite",
    ],
    copts = tflite_copts(),
     linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),

    linkstatic = 1,
)


cc_binary(
    name = "libcnnwrapperx86.so",
    srcs = [
        "cnn.cc",
        "cnn.h",
        "cnn_wrapper.h",
        "cnn_wrapper.cc",
     
    ],
    #copts = tflite_copts(),
    linkopts = select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-latomic",  # Android 5.0 and later supports only PIE
            "-Wl,-soname,libcnnwrapperx86.so",
            "-llog",
        ],
        "//conditions:default": [],
    }),
    deps = [
       "//tensorflow/lite/kernels:builtin_ops",
      #"//tensorflow/contrib/lite/tools:mutable_op_resolver",

    ],

    linkshared = 1,
)

cc_library(
    name = "liblitepc",
    srcs = ["librnnwrapperx86.so"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name="pc_lite_test",
    srcs = ["tflite_rnntest.cc", "tflite_rnnwrapper.h"],
    deps = [":liblitepc",],
)



#for rnn test
cc_binary(
    name = "test",
    srcs = [
        "char_to_word/tflite_rnn.cc", 
        "char_to_word/tflite_rnn.h",
        "utils/tflite_utils.h",
        "char_to_word/char_encoder.h",
        "test.cc",
     #   "tflite_utils.h",
        "utils/LRUCache11.h",
    ],
    copts = tflite_copts(),
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-latomic",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            #"-Wl,-soname,librnnwrapper.so",
            "-llog",
        ],
        "//conditions:default": [],
    }),
    deps = [
       "//tensorflow/lite/kernels:builtin_ops",
      #"//tensorflow/contrib/lite/tools:mutable_op_resolver",
    ],

    linkstatic = 1,
)


