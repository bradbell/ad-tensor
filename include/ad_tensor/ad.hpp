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
This creates a :ref:`ad_type@constant` from a tensor.
If this thread's tape is recording, a copy of the constant is placed
in the tape.
{xrst_literal ,
    BEGIN_PRIVATE_CTOR, END_PRIVATE_CTOR
}


{xrst_end ad_private}
------------------------------------------------------------------------------
*/
#include <torch/torch.h>
//
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/devel/op_enum.hpp>
//
// BINARY_OP
# define BINARY_OP(op, op_enum) \
    ad_t operator op (const ad_t& rhs) const \
{ return binary( devel::op_enum_t:: op_enum, *this, rhs ); }
//
// recording
namespace ad_tensor{ class recording; }
//
// BEGIN_AD_CLASS
namespace ad_tensor { class ad_t
// END_AD_CLASS
    {
    //
    // recording, record
    friend class recording;
    //
// BEGIN_PRIVATE
private:
    //
    // BEGIN_MEMBER_DATA
    size_t        tape_id_;
    size_t        index_;
    at::Tensor    tensor_;
    ad_type_t     ad_type_;
    // END_MEMBER_DATA
    //
    // BEGIN_PRIVATE_CTOR
    ad_t( size_t tape_id, size_t index, at::Tensor&& tensor, ad_type_t ad_type)
    : tape_id_(tape_id), index_(index), tensor_(tensor), ad_type_(ad_type)
    { }
    // END_PRIVATE_CTOR
    //
    // BEGIN_BINARY
    static ad_t binary(
        devel::op_enum_t op_enum, const ad_t& lhs, const ad_t& rhs
    );
    // END_BINARY
public:
    // BEGIN_PUBLIC_CTOR
    ad_t( at::Tensor&& tensor );
    // END_PUBLIC_CTOR
    //
    // BEGIN_TO_TENSOR
    const at::Tensor& tensor(void) const
    // END_TO_TENSOR
    {   return tensor_; }
    //
    // Binary operators
    BINARY_OP(+, add)
    BINARY_OP(-, sub)
    BINARY_OP(*, mul)
    BINARY_OP(/, div)
}; }
