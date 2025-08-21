/*
 * Copyright (c) 2025 dingodb.com, Inc. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DINGOFS_CLIENT_COMMON_UTILS_H_
#define DINGOFS_CLIENT_COMMON_UTILS_H_

#include <sstream>

namespace dingofs {
namespace client {

static std::string Char2Addr(const char* p) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::nouppercase << reinterpret_cast<uintptr_t>(p);
  return oss.str();
}

static std::string StringToHex(const std::string& str) {
  static const char hex_chars[] = "0123456789abcdef";
  std::string result;
  result.reserve(str.size() * 2);

  for (unsigned char c : str) {
    result += hex_chars[c >> 4];
    result += hex_chars[c & 0x0F];
  }

  return result;
}

}  // namespace client
}  // namespace dingofs

#endif  // DINGOFS_CLIENT_COMMON_UTILS_H_