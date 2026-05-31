// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <cassert>
#include <ad_tensor/dev/broadcast.hpp>
#include <ad_tensor/vector.hpp>
/*
{xrst_begin broadcast dev}
{xrst_spell
    res
}

Determine Broadcast Dimensions for a Result Argument Pair
#########################################################

Prototype
*********
{xrst_literal ,
    // BEGIN_BROADCAST, END_BROADCAST
}

res
***
contains the result of the binary operations.

arg
***
contains one of the left or right argument for the binary operation.

dim
***
The return dim is the dimension indices, in the shape of res,
where arg was broadcast to match the shape of the other argument
to the binary operation.
The return is thread local and not valid for use by other threads.
In addition, it is not valid after the next call to broadcast.

arg
===
It helps think of arg as reshaped, with ones in the initial dimensions,
so that its shape as the same length as res.


{xrst_end broadcast}
*/
// BEGIN_BROADCAST
namespace ad_tensor { namespace dev {
    // dim = broadcast(res, arg)
    c10::ArrayRef<long int> broadcast(
        const at::Tensor& res, const at::Tensor& arg
    )
// END_BROADCAST
{   //
    // dim
    thread_local vector<long int> dim;
#ifdef NDEBUG
    dim.resize(0)
#else
    size_t cap = dim.capacity();
    dim.resize(0);
    assert( dim.capacity() == cap && "binary_broadcast: "
        "std::vector::resize to zero changed capacity"
    );
# endif
    //
    c10::ArrayRef<long int> res_sizes = res.sizes();
    c10::ArrayRef<long int> arg_sizes = arg.sizes();
    if( res_sizes.equals( arg_sizes) ) {
        return c10::ArrayRef<long int> (dim);
    }
    size_t res_len = res_sizes.size();
    size_t arg_len = arg_sizes.size();
    assert( arg_len <= res_len && "binary_broadcast: "
        "arg.sizes() is longer than res.sizes()"
    );
    for(size_t i = 0; i < res_len; ++i)
    {   size_t res_index  = res_len - i - 1;
        if( i < arg_len )
        {   size_t arg_index = arg_len - i - 1;
            if( arg_sizes[arg_index] != res_sizes[res_index] )
            {   assert( arg_sizes == 1 && "binary_broadcast: "
                    "arg size is one one and different from res size"
                );
                dim.push_back( static_cast<long int>(res_index) );
            }
        } else if( 1 < res_sizes[res_index] ) {
            dim.push_back(res_index);
        }
    }
    return c10::ArrayRef<long int> (dim);
} } }
