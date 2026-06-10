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
    rhs
    numel
}

Mode Plus Or Minus Equals For Tensors
#####################################

Prototype
*********
{xrst_literal ,
    BEGIN_PLUS_EQUAL, END_PLUS_EQUAL
    BEGIN_MINUS_EQUAL, END_MINUS_EQUAL
}


rhs
***
this is the right hand side that was are adding to (or subtracting from).
It is assumed that this is non-empty.

target
******
This is the target we are adding to (or subtracting from).
The special case where on input target.numel() is zero
(target is empty) corresponds to a zero target value.

dim
***
this is the dimensions rhs that are summed before adding to
(or subtracting from) target.
if this has size zero (its default value), no such summing is done.

{xrst_end plus_minus_equal}
*/
namespace ad_tensor { namespace dev {
// BEGIN_PLUS_EQUAL
    template <class TensorType> void inline plus_equal(
        TensorType&                    target ,
        const TensorType&              rhs ,
        const c10::IntArrayRef&        dim  = c10::IntArrayRef() )
// END_PLUS_EQUAL
    {   //
        assert( rhs.numel() != 0 );
        if( dim.size() == 0 ) {
            if( target.numel() == 0 ) {
                target = rhs;
            } else {
                target += rhs;
                }
                } else {
            TensorType compress = rhs.sum(dim);
            if( target.numel() == 0 ) {
                target = compress;
            } else {
                target += compress;
            }
        }
    }
// BEGIN_MINUS_EQUAL
    template <class TensorType> void inline minus_equal(
        TensorType&                    target ,
        const TensorType&              rhs ,
        const c10::IntArrayRef&        dim  = c10::IntArrayRef() )
// END_MINUS_EQUAL
    {   //
        assert( rhs.numel() != 0 );
        if( dim.size() == 0 ) {
            if( target.numel() == 0 ) {
                target = - rhs;
            } else {
                target -= rhs;
            }
        } else {
            TensorType compress = rhs.sum(dim);
            if( target.numel() == 0 ) {
                target = - compress;
            } else {
                target -= compress;
            }
        }
    }
} }
