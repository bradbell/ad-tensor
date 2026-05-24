# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
add_executable(tests
    dev/op_enum.cpp
    dev/base_op.cpp
    dev/tape.cpp
)
#
target_link_libraries(tests
    ad_tensor
    GTest::gtest_main
    ${TORCH_LIBRARIES}
)
