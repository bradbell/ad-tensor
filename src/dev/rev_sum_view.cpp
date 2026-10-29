// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <cassert>
#include <ad_tensor/dev/rev_sum_view.hpp>
#include <ad_tensor/vector.hpp>
/*
{xrst_begin rev_sum_view dev}
{xrst_spell
    res
}

Reshape for a Sum Result so it Broadcasts to Argument Shape
###########################################################

Syntax
******
{xrst_code cpp}
    rev_sum_view(dim, res_shape, arg_shape, res_reshape)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_REV_SUM_VIEW, END_REV_SUM_VIEW
}

dim
***
is the dimension argument to the sum operation.

res_shape
*********
contains the shape of the result of the sum operation.

arg_shape
*********
contains the shape of the tensor argument to the sum operation.

res_reshape
***********
Only the capacity of the input of res_reshape matters
(it determines if new memory is allocated).
Upon return, res_reshape, is the new shape for the result
so the it will properly broadcast to the argument shape.

{xrst_end rev_sum_view}
*/
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// BEGIN_REV_SUM_VIEW
void rev_sum_view(
    const c10::IntArrayRef&    dim         ,
    const c10::IntArrayRef&    res_shape   ,
    const c10::IntArrayRef&    arg_shape   ,
    vector<int64_t>&           res_reshape
)
{   // END_REV_SUM_VIEW
    //
    // in_dim, res_reshape
    thread_local vector<bool> in_dim;
    in_dim.resize(0);
    res_reshape.resize(0);
    //
    // res_len, arg_len
    size_t res_len = res_shape.size();
    size_t arg_len = arg_shape.size();
    assert( res_len < arg_len && "reverse_sum_view: "
        "res shape is longer than arg shape"
    );
    //
    // in_dim
    in_dim.resize(arg_len, false);
    for(size_t i = 0; i < dim.size(); ++i) {
        in_dim[ size_t( dim[i] ) ] = true;
    }
    //
    // res_reshape
    res_reshape.resize(arg_len);
    size_t res_index = res_len;
    for(size_t i = 0; i < arg_len; ++i) {
        size_t arg_index  = arg_len - i - 1;
        if( in_dim[ arg_index ] ) {
            res_reshape[arg_index] = 1;
        } else if( res_index == 0 ) {
            res_reshape[arg_index] = 1;
        } else {
            res_index -= 1;
            res_reshape[arg_index] = res_shape[res_index];
        }
    }
    assert( res_index == 0 );
    //
    return;
}
} } // End ad_tensor::dev
