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
 * Created Date: 20191217
 * Author: lixiaocui
 */

#include "src/mds/nameserver2/metric.h"

namespace curve {
namespace mds {

void SegmentDiscardMetric::OnReceiveDiscardRequest(int64_t size) {
  pendingSegments_ << 1;
  pendingSize_ << size;
}

void SegmentDiscardMetric::OnDiscardFinish(int64_t size) {
  pendingSegments_ << -1;
  pendingSize_ << -size;

  totalCleanedSegments_ << 1;
  totalCleanedSize_ << size;
}

}  // namespace mds
}  // namespace curve
