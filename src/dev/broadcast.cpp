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
    BEGIN_BROADCAST, END_BROADCAST
}

res_shape
*********
contains the shape of the result of the binary operations.
This is not used and should not be in call when lock is false.

arg_shape
*********
contains the shape of the left or right argument for the binary operation.
It helps think of arg as viewed, with ones in the initial dimensions,
so that its shape as the same length as res.

dim
***
The return dim is the dimension indices, in the res_shape,
where arg_shape was broadcast to match the shape of the other argument
to the binary operation.

{xrst_end broadcast}
*/
// BEGIN_BROADCAST
namespace ad_tensor { namespace dev {
    // dim = broadcast(res, arg)
    vector<int64_t>  broadcast(
        const c10::IntArrayRef&    res_shape ,
        const c10::IntArrayRef&    arg_shape
    )
    {   // END_BROADCAST
        //
        // dim
        vector<int64_t> dim;
        //
        // check for case where tensor shapes are equal
        if( res_shape.equals( arg_shape) ) {
            return dim;
        }
        //
        // dim
        size_t res_len = res_shape.size();
        size_t arg_len = arg_shape.size();
        assert( arg_len <= res_len && "broadcast: "
            "arg shape is longer than res shape"
        );
        for(size_t i = 0; i < res_len; ++i)
        {   size_t res_index  = res_len - i - 1;
            if( i < arg_len )
            {   size_t arg_index = arg_len - i - 1;
                if( arg_shape[arg_index] != res_shape[res_index] )
                {   assert( arg_shape[arg_index] == 1 && "broadcast: "
                        "an arg size is different from its res size and not 1"
                    );
                    dim.push_back( static_cast<int64_t>(res_index) );
                }
            } else if( 1 < res_shape[res_index] ) {
                dim.push_back(res_index);
            }
        }
        return dim;
    }
}  }
