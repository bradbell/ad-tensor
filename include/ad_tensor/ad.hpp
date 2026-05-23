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

The AD Tensor Class
###################
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
{xrst_begin ad_private dev}

The ad_t Class Private Elements
###############################

Member Data
***********
{xrst_literal ,
    BEGIN_MEMBER_DATA, END_MEMBER_DATA
}

Constructor
***********
{xrst_literal ,
    BEGIN_PRIVATE_CTOR, END_PRIVATE_CTOR
}


{xrst_end ad_private}
------------------------------------------------------------------------------
*/
#include <torch/torch.h>
//
#include <ad_tensor/ad_type.hpp>
//
// BINARY_OP
# define BINARY_OP(op) \
    ad_t operator op (const ad_t& rhs) const \
    { return ad_t( tensor_ op rhs.tensor_ ); }
//
// record
namespace ad_tensor{ class recording; }
//
// BEGIN_AD_CLASS
namespace ad_tensor { class ad_t
// END_AD_CLASS
    {
    //
    // recording
    friend class recording;
    //
// BEGIN_PRIVATE
private:
    //
    // BEGIN_MEMBER_DATA
    size_t        tape_id_;
    ad_type_t     ad_type_;
    size_t        index_;
    at::Tensor    tensor_;
    // END_MEMBER_DATA
    //
    // BEGIN_PRIVATE_CTOR
    ad_t( size_t tape_id, ad_type_t ad_type, size_t index, at::Tensor&& tensor)
    : tape_id_(tape_id), ad_type_(ad_type), index_(index), tensor_(tensor)
    { }
    // END_PRIVATE_CTOR

public:
    // BEGIN_PUBLIC_CTOR
    ad_t( at::Tensor&& tensor )
    // END_PUBLIC_CTOR
    : tape_id_(0), ad_type_(ad_type_t::constant), tensor_(tensor)
    { }
    // BEGIN_TO_TENSOR
    const at::Tensor& tensor(void) const
    // END_TO_TENSOR
    {   return tensor_; }
    //
    // Binary operators
    BINARY_OP(+)
    BINARY_OP(-)
    BINARY_OP(*)
    BINARY_OP(/)
}; }
