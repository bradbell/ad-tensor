# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
add_library(ad_tensor SHARED
    # BEGIN_SORT_THIS_LINE_PLUS_1
    adten/binary.cpp
    adten/ctor.cpp
    adten/minus.cpp
    adten/recording.cpp
    adten/sum.cpp
    adten/view.cpp
    adfn/forward_der.cpp
    adfn/forward_par.cpp
    adfn/forward_var.cpp
    adfn/get_range.cpp
    adfn/reverse_der.cpp
    dev/broadcast.cpp
    dev/derive_op/add_op.cpp
    dev/derive_op/div_op.cpp
    dev/derive_op/dom_op.cpp
    dev/derive_op/minus_op.cpp
    dev/derive_op/mul_op.cpp
    dev/derive_op/op_enum2derive_op.cpp
    dev/derive_op/sub_op.cpp
    dev/derive_op/sum_op.cpp
    dev/derive_op/view_op.cpp
    dev/get_option.cpp
    dev/rev_sum_view.cpp
    dev/size_ptr2array_ref.cpp
    dev/tape.cpp
    dev/tensor_at_index.cpp
    dev/to_string.cpp
    # END_SORT_THIS_LINE_MINUS_1
)
