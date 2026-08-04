# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
# BEGIN_SORT_THIS_LINE_PLUS_2
add_executable(benchmarks
    fit_poly.cpp
)
# END_SORT_THIS_LINE_MINUS_2
#
target_link_libraries(benchmarks
    ad_tensor
    GTest::gtest_main
    ${TORCH_LIBRARIES}
)
