// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <cassert>
#include <ad_tensor/dev/rev_sum_reshape.hpp>
#include <ad_tensor/vector.hpp>
/*
{xrst_begin rev_sum_reshape dev}
{xrst_spell
    res
}

Determine Reshape for a Sum Result it Broadcast to Argument Sizes
#################################################################

Prototype
*********
{xrst_literal ,
    include/ad_tensor/dev/rev_sum_reshape.hpp
    BEGIN_REV_SUM_RESHAPE, END_REV_SUM_RESHAPE
}

lock
****
Calls to rev_sum_reshape must be made in pairs.
The first (second) call should have lock true (lock false).

dim
***
is the dimension argument to the sum operation.
This is not used and should not be in call when lock is false.

res
***
contains the result of the sum operation.
This is not used and should not be in call when lock is false.

arg
***
contains one of the tensor argument to the sum operation.
This is not used and should not be in call when lock is false.

shape
*****
The return value, shape, is the new shape for the result res
so the it will properly broadcast to the argument shape.
The return is thread local and not valid for use by other threads.
In addition, it only valid until the following call to rev_sum_reshape
with lock false.


{xrst_end rev_sum_reshape}
*/
namespace ad_tensor { namespace dev { // BEGIN_NAMESPACE_AD_TENSOR_DEV
c10::ArrayRef<long> rev_sum_reshape(
    bool                       lock,
    const c10::ArrayRef<long>& dim ,
    const at::Tensor&          res  ,
    const at::Tensor&          arg  ) {
    //
    // locked
    thread_local bool locked = false;
    //
    // in_dim, shape
    thread_local vector<bool> in_dim;
    thread_local vector<long> shape;
    //
    // locked
    if( ! lock )
    {   assert( locked && "reverse_sum_reshape: "
            "a call with lock false was not preceded by a call with lock true"
        );
        locked = false;
        return c10::ArrayRef<long>();
    }
    assert( ! locked && "reverse_sum_reshape: "
        "attempt to get a lock while another call is holding its lock"
    );
    locked = true;
    //
    // res_shape, arg_shape
    c10::ArrayRef<long> res_shape = res.sizes();
    c10::ArrayRef<long> arg_shape = arg.sizes();
    //
    // res_len, arg_len
    size_t res_len = res_shape.size();
    size_t arg_len = arg_shape.size();
    if( res_len == arg_len ) {
        return arg_shape;
    }
    assert( res_len < arg_len && "reverse_sum_reshape: "
        "res shape is longer than arg shape"
    );
    //
    // in_dim
    in_dim.resize(arg_len, false);
    for(size_t i = 0; i < dim.size(); ++i) {
        in_dim[ dim[i] ] = true;
    }
    //
    // shape
    shape.resize(arg_len);
    size_t res_index = res_len;
    for(size_t i = 0; i < arg_len; ++i) {
        size_t arg_index  = arg_len - i - 1;
        if( in_dim[ arg_index ] ) {
            shape[arg_index] = 1;
        } else if( res_index == 0 ) {
            shape[arg_index] = 1;
        } else {
            res_index -= 1;
            shape[arg_index] = res_shape[res_index];
        }
    }
    assert( res_index == 0 );
    //
    return c10::ArrayRef<long> (shape);
}
} } // END_NAMESPACE_AD_TENSOR_DEV
