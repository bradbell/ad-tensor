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

Determine Reshape for a Sum Result it Broadcast to Argument Sizes
#################################################################

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

new_shape
*********
The return value, new_shape, is the new shape for the result res
so the it will properly broadcast to the argument shape.

{xrst_end rev_sum_view}
*/
namespace ad_tensor { namespace dev { // BEGIN_NAMESPACE_AD_TENSOR_DEV
// BEGIN_REV_SUM_VIEW
// new_shape = rev_sum_view(dim, res_shape, arg_shape)
vector<int64_t> rev_sum_view(
    const c10::IntArrayRef&    dim       ,
    const c10::IntArrayRef&    res_shape ,
    const c10::IntArrayRef&    arg_shape
)
{   // END_REV_SUM_VIEW
    //
    // in_dim, new_shape
    vector<bool> in_dim;
    vector<int64_t> new_shape;
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
        in_dim[ dim[i] ] = true;
    }
    //
    // new_shape
    new_shape.resize(arg_len);
    size_t res_index = res_len;
    for(size_t i = 0; i < arg_len; ++i) {
        size_t arg_index  = arg_len - i - 1;
        if( in_dim[ arg_index ] ) {
            new_shape[arg_index] = 1;
        } else if( res_index == 0 ) {
            new_shape[arg_index] = 1;
        } else {
            res_index -= 1;
            new_shape[arg_index] = res_shape[res_index];
        }
    }
    assert( res_index == 0 );
    //
    return new_shape;
}
} } // END_NAMESPACE_AD_TENSOR_DEV
