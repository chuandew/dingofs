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

load("//:copts.bzl", "BASE_FLAGS", "CURVE_DEFAULT_COPTS")

cc_binary(
        name = "curvefsTool",
        srcs = ["curvefsTool.cpp",
                "curvefsTool.h"],
        deps = [
                "//proto:topology_cc_proto",
                "//src/mds/common:mds_common",
                "//src/common:curve_common",
                "@gflags//:gflags",
                "@glog//:glog",
                "@brpc//:brpc",
                "@jsoncpp//:jsoncpp",
        ],
        copts = CURVE_DEFAULT_COPTS,
)
