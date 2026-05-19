# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
add_executable(examples
    ad.cpp
    record.cpp
)
#
target_link_libraries(examples GTest::gtest_main ${TORCH_LIBRARIES})
