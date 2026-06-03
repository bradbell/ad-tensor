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
    include/ad_tensor/dev/broadcast.hpp
    BEGIN_BROADCAST, END_BROADCAST
}

lock
****
Calls to broadcast must be made in pairs.
The first (second) call should have lock true (lock false).

res_shape
*********
contains the shape of the result of the binary operations.
This is not used and should not be in call when lock is false.

arg
***
contains the shape of the left or right argument for the binary operation.
This is not used and should not be in call when lock is false.

dim
***
The return dim is the dimension indices, in the shape of res,
where arg was broadcast to match the shape of the other argument
to the binary operation.
The return is thread local and not valid for use by other threads.
In addition, it only valid until the following call to broadcast
with lock false.

arg
===
It helps think of arg as viewed, with ones in the initial dimensions,
so that its shape as the same length as res.


{xrst_end broadcast}
*/
namespace ad_tensor { namespace dev {
    // dim = broadcast(res, arg)
    c10::ArrayRef<long> broadcast(
        bool                       lock      ,
        const c10::ArrayRef<long>& res_shape ,
        const c10::ArrayRef<long>& arg_shape
    ) {
        //
        // locked
        thread_local bool locked = false;
        //
        // dim
        thread_local vector<long> dim;
        //
        // locked
        if( ! lock )
        {   assert( locked && "broadcast: a call with lock false "
                "was not preceded by a call with lock true"
            );
            locked = false;
            return c10::ArrayRef<long>();
        }
        assert( ! locked && "broadcast: "
            "attempt to get a lock while another call is holding its lock"
        );
        locked = true;
        //
        //
        // dim
#ifdef NDEBUG
        dim.resize(0)
#else
        size_t cap = dim.capacity();
        dim.resize(0);
        assert( dim.capacity() == cap && "broadcast: "
            "std::vector::resize to zero changed capacity"
        );
# endif
        //
        // check for case where tensor shapes are equal
        if( res_shape.equals( arg_shape) ) {
            return c10::ArrayRef<long> ();
        }
        //
        // dim
        size_t res_len = res_shape.size();
        size_t arg_len = arg_shape.size();
        assert( arg_len <= res_len && "binary_broadcast: "
            "arg shape is longer than res shape"
        );
        for(size_t i = 0; i < res_len; ++i)
        {   size_t res_index  = res_len - i - 1;
            if( i < arg_len )
            {   size_t arg_index = arg_len - i - 1;
                if( arg_shape[arg_index] != res_shape[res_index] )
                {   assert( arg_shape[arg_index] == 1 && "binary_broadcast: "
                        "an arg size is different from its res size and not 1"
                    );
                    dim.push_back( static_cast<long>(res_index) );
                }
            } else if( 1 < res_shape[res_index] ) {
                dim.push_back(res_index);
            }
        }
        return c10::ArrayRef<long> (dim);
    }
}  }
