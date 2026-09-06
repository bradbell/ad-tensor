#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <ad_tensor/no_elements.hpp>
//
/*
{xrst_begin plus_minus_equal dev}
{xrst_spell
    rhs
}

Mode Plus Or Minus Equals For Tensors
#####################################

Prototype
*********
{xrst_literal ,
    BEGIN_PLUS_EQUAL, END_PLUS_EQUAL
    BEGIN_MINUS_EQUAL, END_MINUS_EQUAL
}


target
******
This is the target we are adding to (or subtracting from).
The special case where on input has_elements(target) is false
corresponds to a zero target value.

rhs
***
this is the right hand side that was are adding to (or subtracting from).
The special case where has_elements(rhs) is false
corresponds to a zero right hand side.
In this case, target does not change.

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
        if( no_elements(rhs) ) {
            return;
        }
        if( dim.size() == 0 ) {
            if( no_elements(target) ) {
                target = rhs.clone();
            } else {
                target += rhs;
                }
                } else {
            TensorType compress = rhs.sum(dim);
            if( no_elements(target) ) {
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
        if( no_elements(rhs) ) {
            return;
        }
        if( dim.size() == 0 ) {
            if( no_elements(target) ) {
                target = - rhs;
            } else {
                target -= rhs;
            }
        } else {
            TensorType compress = rhs.sum(dim);
            if( no_elements(target) ) {
                target = - compress;
            } else {
                target -= compress;
            }
        }
    }
} }
