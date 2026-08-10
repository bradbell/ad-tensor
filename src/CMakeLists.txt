# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
add_library(ad_tensor SHARED
    # BEGIN_SORT_THIS_LINE_PLUS_1
    adfn/forward_dep.cpp
    adfn/forward_der.cpp
    adfn/forward_par.cpp
    adfn/forward_var.cpp
    adfn/get_range.cpp
    adfn/optimize.cpp
    adfn/print_con.cpp
    adfn/reverse_der.cpp
    adten/binary.cpp
    adten/call_atom.cpp
    adten/ctor.cpp
    adten/matmul.cpp
    adten/minus.cpp
    adten/recording.cpp
    adten/solve.cpp
    adten/sum.cpp
    adten/transpose.cpp
    adten/unary.cpp
    adten/view.cpp
    adten/where.cpp
    atom.cpp
    base_atom.cpp
    chkpnt.cpp
    dev/atom.cpp
    dev/broadcast.cpp
    dev/chkpnt.cpp
    dev/derive_op/add_op.cpp
    dev/derive_op/call_op.cpp
    dev/derive_op/call_result_op.cpp
    dev/derive_op/div_op.cpp
    dev/derive_op/dom_op.cpp
    dev/derive_op/eq_op.cpp
    dev/derive_op/exp_op.cpp
    dev/derive_op/ge_op.cpp
    dev/derive_op/gt_op.cpp
    dev/derive_op/inverse_op.cpp
    dev/derive_op/le_op.cpp
    dev/derive_op/logdet_op.cpp
    dev/derive_op/lt_op.cpp
    dev/derive_op/matmul_op.cpp
    dev/derive_op/minus_op.cpp
    dev/derive_op/mul_op.cpp
    dev/derive_op/ne_op.cpp
    dev/derive_op/op_enum2derive_op.cpp
    dev/derive_op/solve_op.cpp
    dev/derive_op/sub_op.cpp
    dev/derive_op/sum_op.cpp
    dev/derive_op/transpose_op.cpp
    dev/derive_op/view_op.cpp
    dev/derive_op/where_op.cpp
    dev/optimize/new_agraph.cpp
    dev/optimize/new_call.cpp
    dev/optimize/new_operator.cpp
    dev/optimize/old2new.cpp
    dev/optimize/optimize_con.cpp
    dev/optimize/rng_depend.cpp
    dev/rev_sum_view.cpp
    dev/tape.cpp
    dev/tensor_at_index.cpp
    dev/to_string.cpp
    dev/vec_sets.cpp
    sparsity.cpp
    to_string.cpp
    # END_SORT_THIS_LINE_MINUS_1
)
target_link_libraries( ad_tensor ${TORCH_LIBRARIES} )
