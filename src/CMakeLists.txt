# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
add_library(ad_tensor SHARED
    # BEGIN_SORT_THIS_LINE_PLUS_1
    ad/ctor.cpp
    ad/recording.cpp
    adfn/forward_par.cpp
    dev/ad.cpp
    dev/op_derive/add_op.cpp
    dev/op_derive/div_op.cpp
    dev/op_derive/dom_op.cpp
    dev/op_derive/mul_op.cpp
    dev/op_derive/op_enum2base_op.cpp
    dev/op_derive/sub_op.cpp
    dev/tape.cpp
    dev/to_string.cpp
    # END_SORT_THIS_LINE_MINUS_1
)
