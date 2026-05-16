# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
add_executable(tests
    ad.cpp
    #
    devel/op_enum.cpp
    devel/op_base.cpp
    devel/tape.cpp
)
#
target_link_libraries(tests GTest::gtest_main ${TORCH_LIBRARIES})
