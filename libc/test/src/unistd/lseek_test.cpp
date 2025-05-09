//===-- Unittests for lseek -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/lseek.h"
#include "src/unistd/read.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <unistd.h>

using LlvmLibcUniStd = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcUniStd, LseekTest) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *FILENAME = "testdata/lseek.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_RDONLY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  constexpr const char LSEEK_TEST[] = "lseek test";
  constexpr ssize_t LSEEK_TEST_SIZE = sizeof(LSEEK_TEST) - 1;

  char read_buf[20];
  ASSERT_THAT(LIBC_NAMESPACE::read(fd, read_buf, LSEEK_TEST_SIZE),
              Succeeds(LSEEK_TEST_SIZE));
  read_buf[LSEEK_TEST_SIZE] = '\0';
  EXPECT_STREQ(read_buf, LSEEK_TEST);

  // Seek to the beginning of the file and re-read.
  ASSERT_THAT(LIBC_NAMESPACE::lseek(fd, 0, SEEK_SET), Succeeds(off_t(0)));
  ASSERT_THAT(LIBC_NAMESPACE::read(fd, read_buf, LSEEK_TEST_SIZE),
              Succeeds(LSEEK_TEST_SIZE));
  read_buf[LSEEK_TEST_SIZE] = '\0';
  EXPECT_STREQ(read_buf, LSEEK_TEST);

  // Seek to the beginning of the file from the end and re-read.
  ASSERT_THAT(LIBC_NAMESPACE::lseek(fd, -LSEEK_TEST_SIZE, SEEK_END),
              Succeeds(off_t(0)));
  ASSERT_THAT(LIBC_NAMESPACE::read(fd, read_buf, LSEEK_TEST_SIZE),
              Succeeds(LSEEK_TEST_SIZE));
  read_buf[LSEEK_TEST_SIZE] = '\0';
  EXPECT_STREQ(read_buf, LSEEK_TEST);

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcUniStd, LseekFailsTest) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *FILENAME = "testdata/lseek.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_RDONLY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  EXPECT_THAT(LIBC_NAMESPACE::lseek(fd, -1, SEEK_CUR), Fails<off_t>(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}
