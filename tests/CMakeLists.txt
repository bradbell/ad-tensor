# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
# BEGIN_SORT_THIS_LINE_PLUS_2
add_executable(tests
    adfn/reverse_der.cpp
    adten/matmul_op.cpp
    adten/minus_op.cpp
    adten/solve_op.cpp
    adten/sum_op.cpp
    adten/transpose_op.cpp
    adten/view_op.cpp
    dev/base_op.cpp
    dev/op_enum.cpp
    dev/tape.cpp
    dev/vec_set.cpp
)
# END_SORT_THIS_LINE_MINUS_2
#
target_link_libraries(tests
    ad_tensor
    GTest::gtest_main
    ${TORCH_LIBRARIES}
)
