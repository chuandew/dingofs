#
#  Copyright (c) 2020 NetEase Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

load("//:copts.bzl", "CURVE_DEFAULT_COPTS")

cc_library(
    name = "bitmap",
    srcs = [
        "bitmap.cpp",
    ],
    hdrs = [
        "bitmap.h",
    ],
    copts = CURVE_DEFAULT_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "@glog//:glog",
    ],
)

cc_library(
    name = "curve_common",
    srcs = glob(
        [
            "*.h",
            "*.cpp",
        ],
        exclude = [
            "authenticator.*",
            "s3_adapter.*",
            "snapshotclone_define.*",
            "macros.h",
        ],
    ),
    copts = CURVE_DEFAULT_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//src/common/concurrent:curve_concurrent",
        ":macros",
        "@brpc//:brpc",
        "@glog//:glog",
        "@uuid//:uuid",
    ],
)

cc_library(
    name = "curve_auth",
    srcs = glob([
        "authenticator.h",
        "authenticator.cpp",
    ]),
    copts = CURVE_DEFAULT_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//src/common/concurrent:curve_concurrent",
        "@brpc//:brpc",
        "@glog//:glog",
    ],
)

cc_library(
    name = "curve_uncopy",
    srcs = glob([
        "uncopyable.h",
    ]),
    copts = CURVE_DEFAULT_COPTS,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "curve_s3_adapter",
    srcs = glob([
        "s3_adapter.h",
        "s3_adapter.cpp",
    ]),
    copts = CURVE_DEFAULT_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//src/common:curve_common",
        "@glog//:glog",
        "@aws//:aws",
    ],
)

cc_library(
    name = "curve_snapshotclone",
    srcs = glob([
        "snapshotclone_define.*",
    ]),
    copts = CURVE_DEFAULT_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "@jsoncpp//:jsoncpp",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    srcs = [],
    visibility = ["//visibility:public"],
    copts = CURVE_DEFAULT_COPTS,
)
