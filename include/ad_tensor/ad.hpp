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
    BEGIN_PUBLIC_CTOR, END_PUBLIC_CTOR
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
#include <torch/torch.h>
//
#include <ad_tensor/devel/ad_type.hpp>
//
// BINARY_OP
# define BINARY_OP(op) \
    ad_t operator op (const ad_t& rhs) const \
    { return ad_t( tensor_ op rhs.tensor_ ); }
//
// record
namespace ad_tensor{ class record_t; }
//
// BEGIN_AD_CLASS
namespace ad_tensor { class ad_t
// END_AD_CLASS
    {
    //
    // record_t
    friend class record_t;
    //
private:
    // BEGIN_PRIVATE_DATA
    size_t           tape_id_;
    devel::ad_type_t ad_type_;
    torch::Tensor    tensor_;
    // END_PRIVATE_DATA

    // BEGIN_PRIVATE_CTOR
    ad_t(size_t tape_id, devel::ad_type_t ad_type, torch::Tensor&& tensor)
    : tape_id_(tape_id), ad_type_(ad_type), tensor_(tensor)
    { }
    // END_PRIVATE_CTOR

public:
    // BEGIN_PUBLIC_CTOR
    ad_t( torch::Tensor&& tensor )
    // END_PUBLIC_CTOR
    : tape_id_(0), ad_type_(devel::ad_type_t::constant), tensor_(tensor)
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
