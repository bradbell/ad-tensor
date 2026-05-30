# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
# BEGIN_SORT_THIS_LINE_PLUS_2
add_executable(examples
    ad/binary.cpp
    ad/recording.cpp
    adfn/forward_der.cpp
    adfn/forward_par.cpp
    adfn/forward_var.cpp
    adfn/get_range.cpp
    adfn/reverse_der.cpp
)
# END_SORT_THIS_LINE_MINUS_2
#
target_link_libraries(examples
    ad_tensor
    GTest::gtest_main
    ${TORCH_LIBRARIES}
)
