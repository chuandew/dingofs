/*
 *  Copyright (c) 2020 NetEase Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
 * Project: curve
 * Created Date: Friday October 18th 2019
 * Author: yangyaokai
 */

#ifndef TEST_INTEGRATION_COMMON_CONFIG_GENERATOR_H_
#define TEST_INTEGRATION_COMMON_CONFIG_GENERATOR_H_

#include <string>

#include "src/common/configuration.h"
#include "test/util/config_generator.h"

namespace curve {

#define DEFAULT_LOG_DIR "./runlog/"

using curve::common::Configuration;

// chunkserver test config
class CSTConfigGenerator : public ConfigGenerator {
 public:
  CSTConfigGenerator() {}
  ~CSTConfigGenerator() {}
  bool Init(const std::string& port) {
    // 加载配置文件模板
    config_.SetConfigPath(DEFAULT_CHUNKSERVER_CONF);
    if (!config_.LoadConfig()) {
      return false;
    }
    SetKV("global.port", port);

    std::string stor_uri = "local://./" + port + "/";
    std::string meta_uri = stor_uri + "chunkserver.dat";
    std::string data_uri = stor_uri + "copysets";
    std::string recycler_uri = stor_uri + "recycler";
    SetKV("chunkserver.stor_uri", stor_uri);
    SetKV("chunkserver.meta_uri", meta_uri);
    SetKV("copyset.chunk_data_uri", data_uri);
    SetKV("copyset.raft_log_uri", data_uri);
    SetKV("copyset.raft_meta_uri", data_uri);
    SetKV("copyset.raft_snapshot_uri", data_uri);
    SetKV("copyset.recycler_uri", recycler_uri);

    std::string cfpoolDir = "./" + port + "/chunkfilepool/";
    std::string cfpoolMetaPath = "./" + port + "/chunkfilepool.meta";
    SetKV("chunkfilepool.chunk_file_pool_dir", cfpoolDir);
    SetKV("chunkfilepool.meta_path", cfpoolMetaPath);

    SetKV("chunkfilepool.enable_get_chunk_from_pool", "false");
    std::string walPoolDir = "./" + port + "/walfilepool/";
    std::string walPoolMetaPath = "./" + port + "/walfilepool.meta";
    SetKV("walfilepool.file_pool_dir", walPoolDir);
    SetKV("walfilepool.meta_path", walPoolMetaPath);
    SetKV("walfilepool.enable_get_segment_from_pool", "false");
    SetKV("walfilepool.use_chunk_file_pool", "false");

    SetKV("chunkserver.common.logDir", DEFAULT_LOG_DIR);

    configPath_ = "./" + port + "/chunkserver.conf";
    config_.SetConfigPath(configPath_);
    return true;
  }
};

}  // namespace curve

#endif  // TEST_INTEGRATION_COMMON_CONFIG_GENERATOR_H_
