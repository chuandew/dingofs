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
 * Created Date: Thur March 28th 2019
 * Author: lixiaocui
 */

#include "src/mds/nameserver2/idgenerator/inode_id_generator.h"

#include <glog/logging.h>

#include <string>

#include "src/common/string_util.h"
#include "src/mds/nameserver2/helper/namespace_helper.h"

namespace curve {
namespace mds {

bool InodeIdGeneratorImp::GenInodeID(InodeID* id) {
  return generator_->GenID(id);
}

}  // namespace mds
}  // namespace curve
