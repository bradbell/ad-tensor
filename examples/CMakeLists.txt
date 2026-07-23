# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
# BEGIN_SORT_THIS_LINE_PLUS_2
add_executable(examples
    adfn/forward_dep.cpp
    adfn/forward_der.cpp
    adfn/forward_par.cpp
    adfn/forward_var.cpp
    adfn/get_range.cpp
    adfn/get_started.cpp
    adfn/optimize.cpp
    adfn/reverse_der.cpp
    adten/binary.cpp
    adten/matmul.cpp
    adten/minus.cpp
    adten/recording.cpp
    adten/solve.cpp
    adten/sum.cpp
    adten/transpose.cpp
    adten/view.cpp
    atom/ad_forward_der.cpp
    atom/ad_reverse_der.cpp
    atom/get_started.cpp
    chkpnt/ad_forward_der.cpp
    chkpnt/ad_reverse_der.cpp
    chkpnt/get_started.cpp
)
# END_SORT_THIS_LINE_MINUS_2
#
target_link_libraries(examples
    ad_tensor
    GTest::gtest_main
    ${TORCH_LIBRARIES}
)
