#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
/*
------------------------------------------------------------------------------
{xrst_begin ad usr}
{xrst_spell
    op
    rhs
}

The ad_t class
##############
{xrst_literal ,
    BEGIN_AD_CLASS, END_AD_CLASS
}

From Tensor
***********
{xrst_literal ,
    BEGIN_FROM_TENSOR, END_FROM_TENSOR
}

To Tensor
*********
{xrst_literal ,
    BEGIN_TO_TENSOR, END_TO_TENSOR
}

Binary Operators
****************
For *op* equal ``+``, ``-``, ``*``, ``/`` :
{xrst_code cpp}
    ad_t operator op (const ad_t& rhs) const
{xrst_code}

Example
*******
{xrst_literal ,
    examples/ad.cpp
    // BEGIN_CPP, // END_CPP
}

{xrst_end ad}
------------------------------------------------------------------------------
*/
# include <torch/torch.h>
//
// BINARY_OP
# define BINARY_OP(op) \
    ad_t operator op (const ad_t& rhs) const \
    { return ad_t( tensor_ op rhs.tensor_ ); }
//
// BEGIN_AD_CLASS
namespace ad_tensor { class ad_t
// END_AD_CLASS
{
private:
    // tensor_
    torch::Tensor tensor_;

public:
    // BEGIN_FROM_TENSOR
    ad_t( torch::Tensor&& tensor )
    // END_FROM_TENSOR
    : tensor_(tensor)
    { }
    // BEGIN_TO_TENSOR
    const torch::Tensor& tensor(void) const
    // END_TO_TENSOR
    {   return tensor_; }
    //
    // Binary operators
    BINARY_OP(+)
    BINARY_OP(-)
    BINARY_OP(*)
    BINARY_OP(/)
}; }
