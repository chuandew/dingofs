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

cc_test(
    name = "chunkserver-concurrent-test",
    srcs = glob([
        "peer_cluster.h",
        "peer_cluster.cpp",
        "chunkserver_concurrent_test.cpp",
    ]),
    copts = CURVE_TEST_COPTS,
    deps = [
        "//proto:chunkserver-cc-protos",
        "//src/chunkserver:chunkserver-test-lib",
        "//src/common:curve_common",
        "//test/chunkserver/datastore:filepool_helper",
        "//test/integration/common:integration-test-common",
        "@braft//:braft",
        "@brpc//:brpc",
        "@gflags//:gflags",
        "@glog//:glog",
        "@leveldb//:leveldb",
        "@gtest//:gtest",
    ],
)

cc_test(
    name = "chunkserver-basic-test",
    srcs = glob([
        "peer_cluster.h",
        "peer_cluster.cpp",
        "chunkserver_basic_test.cpp",
    ]),
    copts = CURVE_TEST_COPTS,
    deps = [
        "//proto:chunkserver-cc-protos",
        "//src/chunkserver:chunkserver-test-lib",
        "//src/common:curve_common",
        "//test/chunkserver/datastore:filepool_helper",
        "//test/integration/common:integration-test-common",
        "@braft//:braft",
        "@brpc//:brpc",
        "@gflags//:gflags",
        "@glog//:glog",
        "@leveldb//:leveldb",
        "@gtest//:gtest",
    ],
)

cc_test(
    name = "chunkserver-clone-recover-test",
    srcs = glob([
        "peer_cluster.h",
        "peer_cluster.cpp",
        "chunkserver_clone_recover.cpp",
    ]),
    copts = CURVE_TEST_COPTS,
    deps = [
        "//proto:chunkserver-cc-protos",
        "//src/chunkserver:chunkserver-test-lib",
        "//src/common:curve_common",
        "//test/chunkserver/datastore:filepool_helper",
        "//test/integration/common:integration-test-common",
        "//test/integration/client/common:client_integration_common_lib",
        "//test/integration/cluster_common:integration_cluster_common",
        "@braft//:braft",
        "@brpc//:brpc",
        "@gflags//:gflags",
        "@glog//:glog",
        "@leveldb//:leveldb",
        "@gtest//:gtest",
    ],
)
