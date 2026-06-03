#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
//
/*
{xrst_begin plus_minus_equal dev}
{xrst_spell
    res
    numel
}

Reverse Mode Plus Or Minus Equals For Tensors
#############################################

Prototype
*********
{xrst_literal ,
    BEGIN_REV_PLUS_EQUAL, END_REV_PLUS_EQUAL
    BEGIN_REV_MINUS_EQUAL, END_REV_MINUS_EQUAL
}

dim
***
this is the dimensions that arg was broadcast over to get the result shape;
see :ref:`broadcast-name` .

res
***
this is the result value that was are adding to the argument.
It is assume that this no non-empty because there is not reason to
process an empty result during reverse mode.

arg
***
This is the argument. The special case where arg.numel() is zero
(arg is empty) corresponds to a zero argument value.


{xrst_end plus_minus_equal}
*/
// BEGIN_REV_PLUS_EQUAL
namespace ad_tensor { namespace dev {
    template <class TensorType> void inline rev_plus_equal(
        const c10::ArrayRef<long>&     dim ,
        const TensorType&              res ,
        TensorType&                    arg )
// END_REV_PLUS_EQUAL
    {   //
        assert( res.numel() != 0 );
        if( dim.size() == 0 ) {
            if( arg.numel() == 0 ) {
                arg = res;
            } else {
                arg += res;
            }
        } else {
            TensorType compress = res.sum(dim);
            if( arg.numel() == 0 ) {
                arg = compress;
            } else {
                arg += compress;
            }
        }
    }
// BEGIN_REV_MINUS_EQUAL
    template <class TensorType> void inline rev_minus_equal(
        const c10::ArrayRef<long>&     dim ,
        const TensorType&              res ,
        TensorType&                    arg )
// END_REV_MINUS_EQUAL
    {   //
        assert( res.numel() != 0 );
        if( dim.size() == 0 ) {
            if( arg.numel() == 0 ) {
                arg = - res;
            } else {
                arg -= res;
            }
        } else {
            TensorType compress = res.sum(dim);
            if( arg.numel() == 0 ) {
                arg = - compress;
            } else {
                arg -= compress;
            }
        }
    }
} }
