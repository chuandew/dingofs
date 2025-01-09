/*
 * Copyright (c) 2024 dingodb.com, Inc. All Rights Reserved
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

/*
 * Project: DingoFS
 * Created Date: 2024-09-04
 * Author: Jingli Chen (Wine93)
 */

#include <cstdlib>

#include "base/filepath/filepath.h"
#include "client/blockcache/local_filesystem.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "utils/uuid.h"

namespace dingofs {
namespace client {
namespace blockcache {

using ::dingofs::utils::UUIDGenerator;
using ::dingofs::base::filepath::PathJoin;
using FileInfo = LocalFileSystem::FileInfo;

class LocalFileSystemTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_dir_ = "." + UUIDGenerator().GenerateUUID();
    std::system(("mkdir -p " + root_dir_).c_str());
  }

  void TearDown() override { std::system(("rm -rf " + root_dir_).c_str()); }

 protected:
  std::string root_dir_;
};

TEST_F(LocalFileSystemTest, MkDirs) {
  auto fs = std::make_unique<LocalFileSystem>();

  auto rc = fs->MkDirs(PathJoin({root_dir_, "a", "b"}));
  ASSERT_EQ(rc, BCACHE_ERROR::OK);

  std::string path = PathJoin({root_dir_, "a", "b", "file1"});
  rc = fs->WriteFile(path, "hello world", 11);
  ASSERT_EQ(rc, BCACHE_ERROR::OK);

  size_t count;
  std::shared_ptr<char> buffer;
  rc = fs->ReadFile(path, buffer, &count);
  ASSERT_EQ(rc, BCACHE_ERROR::OK);
  ASSERT_EQ(count, 11);
  ASSERT_EQ(std::string(buffer.get(), count), "hello world");
}

TEST_F(LocalFileSystemTest, Walk) {
  auto fs = std::make_unique<LocalFileSystem>(nullptr);

  ASSERT_EQ(fs->WriteFile(PathJoin({root_dir_, "a"}), "x", 1),
            BCACHE_ERROR::OK);
  ASSERT_EQ(fs->WriteFile(PathJoin({root_dir_, "b"}), "x", 1),
            BCACHE_ERROR::OK);
  ASSERT_EQ(fs->WriteFile(PathJoin({root_dir_, "c"}), "x", 1),
            BCACHE_ERROR::OK);

  std::vector<std::string> files;
  auto rc = fs->Walk(PathJoin({root_dir_}),
                     [&](const std::string& prefix, const FileInfo& info) {
                       files.emplace_back(info.name);
                       return BCACHE_ERROR::OK;
                     });
  ASSERT_EQ(rc, BCACHE_ERROR::OK);
  ASSERT_EQ(files.size(), 3);
  ASSERT_EQ(files[0], "c");
  ASSERT_EQ(files[1], "a");
  ASSERT_EQ(files[2], "b");
}

TEST_F(LocalFileSystemTest, WriteFile) {
  auto fs = std::make_unique<LocalFileSystem>();

  std::string path = PathJoin({root_dir_, "f1"});
  ASSERT_EQ(fs->WriteFile(path, "x", 1), BCACHE_ERROR::OK);

  size_t count;
  std::shared_ptr<char> buffer;
  ASSERT_EQ(fs->ReadFile(path, buffer, &count), BCACHE_ERROR::OK);
  ASSERT_EQ(count, 1);
  ASSERT_EQ(std::string(buffer.get(), count), "x");

  ASSERT_EQ(fs->WriteFile(path, "yy", 2), BCACHE_ERROR::OK);
  ASSERT_EQ(fs->ReadFile(path, buffer, &count), BCACHE_ERROR::OK);
  ASSERT_EQ(count, 2);
  ASSERT_EQ(std::string(buffer.get(), count), "yy");
}

TEST_F(LocalFileSystemTest, RemoveFile) {
  auto fs = std::make_unique<LocalFileSystem>();

  std::string path = PathJoin({root_dir_, "f1"});
  ASSERT_EQ(fs->WriteFile(path, "x", 1), BCACHE_ERROR::OK);
  ASSERT_TRUE(fs->FileExists(path));

  ASSERT_EQ(fs->RemoveFile(path), BCACHE_ERROR::OK);
  ASSERT_FALSE(fs->FileExists(path));
}

TEST_F(LocalFileSystemTest, HardLink) {
  auto fs = std::make_unique<LocalFileSystem>();

  std::string src = PathJoin({root_dir_, "dir", "f1"});
  std::string dest = PathJoin({root_dir_, "dir", "f2"});
  ASSERT_EQ(fs->WriteFile(src, "x", 1), BCACHE_ERROR::OK);
  ASSERT_TRUE(fs->FileExists(src));
  ASSERT_FALSE(fs->FileExists(dest));

  ASSERT_EQ(fs->HardLink(src, dest), BCACHE_ERROR::OK);
  ASSERT_TRUE(fs->FileExists(dest));
}

TEST_F(LocalFileSystemTest, FileExists) {
  auto fs = std::make_unique<LocalFileSystem>();

  std::string dir = PathJoin({root_dir_, "dir"});
  ASSERT_FALSE(fs->FileExists(dir));
  ASSERT_EQ(fs->MkDirs(dir), BCACHE_ERROR::OK);
  ASSERT_FALSE(fs->FileExists(dir));

  std::string path = PathJoin({dir, "file"});
  ASSERT_FALSE(fs->FileExists(path));
  ASSERT_EQ(fs->WriteFile(path, "x", 1), BCACHE_ERROR::OK);
  ASSERT_TRUE(fs->FileExists(path));
}

}  // namespace blockcache
}  // namespace client
}  // namespace dingofs