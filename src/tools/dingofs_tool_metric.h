/*
 *  Copyright (c) 2021 NetEase Inc.
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
 * Project: dingo
 * Created Date: 2021-10-25
 * Author: chengyi01
 */
#ifndef DINGOFS_SRC_TOOLS_DINGOFS_TOOL_METRIC_H_
#define DINGOFS_SRC_TOOLS_DINGOFS_TOOL_METRIC_H_

#include <brpc/channel.h>
#include <json/json.h>

#include <iostream>
#include <string>

#include "tools/dingofs_tool_define.h"

namespace dingofs {
namespace tools {

enum class MetricStatusCode {
  // success
  kOK = 0,
  // metric not found
  kNotFound = -1,
  // other error
  kOtherErr = -2,
};

class MetricClient {
 public:
  static MetricStatusCode GetMetric(const std::string& addr,
                                    const std::string& subUri,
                                    std::string* value);
  static int GetKeyValueFromJson(const std::string& strJson,
                                 const std::string& key, std::string* value);
  static int GetKeyValueFromString(const std::string& str,
                                   const std::string& key, std::string* value);
  static void TrimMetricString(std::string* str);

 protected:
  static const int kHttpCodeNotFound = 404;
};

}  // namespace tools
}  // namespace dingofs

#endif  // DINGOFS_SRC_TOOLS_DINGOFS_TOOL_METRIC_H_
