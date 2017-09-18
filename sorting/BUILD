py_binary(
    name = "sorting_train",
    srcs = ["sorting_train.py"],
    deps = [
        ":sorting_model",
        "//:sinkhorn_ops",
    ],
)

py_binary(
    name = "sorting_eval",
    srcs = ["sorting_eval.py"],
    deps = [
        ":sorting_model",
        "//:sinkhorn_ops",
    ],
)

py_library(
    name = "sorting_model",
    srcs = ["sorting_model.py"],
    deps = [
        "//:optimizer",
        "//:sinkhorn_ops",
    ],
)
