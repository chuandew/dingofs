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

load("//:copts.bzl", "CURVE_DEFAULT_COPTS", "CURVE_TEST_COPTS")

cc_library(
    name = "leader_election",
    srcs = ["leader_election.cpp"],
    hdrs = ["leader_election.h"],
    copts = CURVE_DEFAULT_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//src/kvstorageclient:kvstorage_client",
        "//src/mds/nameserver2/helper",
        "@gflags//:gflags",
        "@glog//:glog",
        "@libfiu//:libfiu",
    ],
)

cc_library(
    name = "leader_election_for_test",
    srcs = ["leader_election.cpp"],
    hdrs = ["leader_election.h"],
    copts = CURVE_TEST_COPTS,
    defines = ["FIU_ENABLE"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/common:curve_common",
        "//src/kvstorageclient:kvstorage_client",
        "@gflags//:gflags",
        "@glog//:glog",
        "@libfiu//:libfiu",
    ],
)
