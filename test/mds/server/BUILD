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

load("//:copts.bzl", "CURVE_TEST_COPTS")

# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_binary
cc_binary(
    name = "mds-test",
    srcs = glob(["*.cpp"]),
    copts = CURVE_TEST_COPTS,
    defines = ["FIU_ENABLE"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/mds/server:mds_for_test",
        "@libfiu//:libfiu",
        "@gtest//:gtest",
    ],
)
