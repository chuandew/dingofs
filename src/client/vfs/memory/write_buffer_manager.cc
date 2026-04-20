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

#include "client/vfs/memory/write_buffer_manager.h"

#include <fmt/format.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstdlib>
#include <mutex>
#include <vector>

#include "common/helper.h"

DEFINE_bool(vfs_write_buffer_bench_pool, false,
            "[bench] serve WriteBufferManager::Allocate from a pre-touched "
            "slab pool so the allocation never hits virgin memory; isolates "
            "per-page new char[page_size] cost and its page-fault storm on "
            "the write hot path. Bench only — pool is lazily initialized on "
            "first call and never grows.");

namespace dingofs {
namespace client {
namespace vfs {

namespace {

// Thread-local slab pool: each thread has its own ring of pages so memcpy to a
// returned page hits cache lines only this thread owns — no cross-core cache
// ping-pong that a shared pool would cause at 32T concurrency.
//
// ThreadSlab is a thread_local vector-plus-counter; the first Allocate() in a
// thread lazily pre-allocates kPagesPerThread slots and pre-touches each page.
// Thread-local ring of pages; slots are allocated lazily on first use so we
// don't pay a pre-touch tax at startup. After the first pass through the ring
// all slots are warm and reused.
constexpr size_t kPagesPerThread = 256;  // 16 MB per thread at page_size=64 KB

struct ThreadSlab {
  std::array<char*, kPagesPerThread> slots{};
  size_t next{0};
};

thread_local ThreadSlab* tls_slab = nullptr;

char* BenchPoolAllocate(size_t page_size) {
  if (tls_slab == nullptr) {
    tls_slab = new ThreadSlab();
  }
  size_t idx = tls_slab->next++ % tls_slab->slots.size();
  char* p = tls_slab->slots[idx];
  if (p == nullptr) {
    p = static_cast<char*>(std::malloc(page_size));
    CHECK(p != nullptr) << "BenchPool malloc failed";
    tls_slab->slots[idx] = p;
  }
  return p;
}

}  // namespace

WriteBufferManager::WriteBufferManager(int64_t total_bytes, int64_t page_size)
    : total_bytes_(total_bytes),
      page_size_(page_size),
      write_buffer_total_bytes_("vfs_write_buffer_total_bytes_", total_bytes),
      write_buffer_used_pages_("vfs_write_buffer_used_pages", UsedPages, this),
      write_buffer_used_bytes_("vfs_write_buffer_used_bytes", UsedBytes, this) {
}

char* WriteBufferManager::Allocate() {
  if (FLAGS_vfs_write_buffer_bench_pool) {
    used_pages_.fetch_add(1);
    return BenchPoolAllocate(page_size_);
  }

  butil::Timer timer;
  timer.start();
  char* page = new char[page_size_];
  timer.stop();

  VLOG(16) << fmt::format("Allocate page at: {} took: <{:.6f}> ms",
                          Helper::Char2Addr(page), timer.u_elapsed(0.0));

  used_pages_.fetch_add(1);
  return page;
}

void WriteBufferManager::DeAllocate(char* page) {
  if (FLAGS_vfs_write_buffer_bench_pool) {
    // Pool owns the memory; noop.
    used_pages_.fetch_sub(1);
    return;
  }

  butil::Timer timer;
  timer.start();
  delete[] page;
  timer.stop();

  VLOG(16) << fmt::format(
      "Deallocating page at: {} allocation took: <{:.6f}> ms",
      Helper::Char2Addr(page), timer.u_elapsed(0.0));

  used_pages_.fetch_sub(1);
}

int64_t WriteBufferManager::GetPageSize() const { return page_size_; }

int64_t WriteBufferManager::GetTotalBytes() const { return total_bytes_; }

int64_t WriteBufferManager::GetUsedBytes() const {
  return page_size_ * used_pages_.load(std::memory_order_relaxed);
}

double WriteBufferManager::GetUsageRatio() const {
  int64_t total = GetTotalBytes();
  if (total == 0) {
    return 0.0;
  }
  return static_cast<double>(GetUsedBytes()) / total;
}

bool WriteBufferManager::IsHighPressure(double threshold) const {
  return GetUsageRatio() >= threshold;
}

}  // namespace vfs
}  // namespace client
}  // namespace dingofs