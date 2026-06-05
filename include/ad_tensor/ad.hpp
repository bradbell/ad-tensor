#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
/*
------------------------------------------------------------------------------
{xrst_begin ad usr}
{xrst_spell
    op
    lhs
    rhs
    numel
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

sizes, numel
************
For *fun* equals ``sizes``, ``numel``,
return the result for the underlying at::Tensor.
{xrst_literal ,
    BEGIN_SIZES, END_SIZES
    BEGIN_NUMEL, END_NUMEL
}

Binary Operators
****************
For *op* equal ``+``, ``-``, ``*``, ``/`` :
{xrst_code cpp}
    lhs op rhs
{xrst_code}
where lhs and rhs are const ad_t reference

Compound Assignment Operators
*****************************
For *op* equal ``+=``, ``-=``, ``*=``, ``/=`` :
{xrst_code cpp}
    lhs op rhs
{xrst_code}
where lhs and rhs are ad_t references and rhs is.

Other Member Functions
**********************
{xrst_toc_table
    src/ad/recording.cpp
}

Example
*******
{xrst_literal ,
    examples/ad/binary.cpp
    BEGIN_CPP, // END_CPP
}

{xrst_end ad}
------------------------------------------------------------------------------
{xrst_begin_parent ad_dev dev}

The ad_t Class Developer Documentation
######################################
{xrst_toc_table after
    src/ad/binary.cpp
    src/ad/minus.cpp
    src/ad/sum.cpp
    src/ad/view.cpp
}

{xrst_end ad_dev}
------------------------------------------------------------------------------
{xrst_begin ad_member dev}

The ad_t Member Data
####################
{xrst_literal ,
    BEGIN_MEMBER_DATA, END_MEMBER_DATA
}

m_tape_id
*********
We say that this AD tensor matches the current recording if
this thread is currently recording and its m_tape_id is the
same as for this object.

m_index
*******
If this thread matches the current recording,
m_index is the index in this threads tape for this AD tensor.

m_ad_type
*********
If this thread matches the current recording,
m_ad_type is the :ref:`ad_type-name` in this threads tape for this AD tensor.

m_tensor
********
is the tensor corresponding to this AD tensor.

{xrst_end ad_member}
------------------------------------------------------------------------------
{xrst_begin ad_ctor_dev dev}

The ad_t Private Constructor
############################
{xrst_literal ,
    BEGIN_PRIVATE_CTOR, END_PRIVATE_CTOR
}

{xrst_end ad_ctor_dev}
------------------------------------------------------------------------------
*/
#include <torch/torch.h>
//
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/dev/op_enum.hpp>
//
// BINARY_OP
# define BINARY_OP(op, op_enum) \
    \
    ad_t operator op (const ad_t& rhs) const \
    { return binary( dev::op_enum_t:: op_enum, *this, rhs ); } \
    \
    ad_t& operator op ## = (const ad_t& rhs) { \
        *this = *this op rhs; \
        return *this; \
    }
//
// BEGIN_AD_CLASS
namespace ad_tensor { class ad_t
// END_AD_CLASS
{
private:
    //
    // BEGIN_MEMBER_DATA
    size_t        m_tape_id;
    size_t        m_index;
    at::Tensor    m_tensor;
    ad_type_t     m_ad_type;
    // END_MEMBER_DATA
    //
    // BEGIN_PRIVATE_CTOR
    ad_t(
        size_t tape_id,
        size_t index,
        const at::Tensor& tensor,
        ad_type_t ad_type )
    : m_tape_id(tape_id), m_index(index), m_tensor(tensor), m_ad_type(ad_type)
    { }
    // END_PRIVATE_CTOR
    //
    static ad_t binary(
        dev::op_enum_t op_enum, const ad_t& lhs, const ad_t& rhs
    );
public:
    // BEGIN_PUBLIC_CTOR
    explicit ad_t( const at::Tensor& tensor );
    // END_PUBLIC_CTOR
    //
    // BEGIN_SIZES
    c10::ArrayRef<long> sizes(void) const
    // END_SIZES
    {   return m_tensor.sizes(); }
    //
    // BEGIN_NUMEL
    long numel(void) const
    // END_NUMEL
    {   return m_tensor.numel(); }
    //
    // BEGIN_TO_TENSOR
    const at::Tensor& tensor(void) const
    // END_TO_TENSOR
    {   return m_tensor; }
    //
    // BEGIN_START_RECORDING
    // auto [adom_par, adom_var] = ad_t::start_recording(dom_par, dom_var)
    static std::tuple< ad_tensor::vector<ad_t>, ad_tensor::vector<ad_t> > start_recording(
        const ad_tensor::vector<at::Tensor>& dom_par,
        const ad_tensor::vector<at::Tensor>& dom_var
    );
    // END_START_RECORDING
    //
    // BEGIN_STOP_RECORDING
    // adfn = ad_t::stop_recording(arnage)
    static adfn_t stop_recording(const ad_tensor::vector<ad_t>& arange);
    // END_STOP_RECORDING
    //
    // Binary operators
    BINARY_OP(+, add)
    BINARY_OP(-, sub)
    BINARY_OP(*, mul)
    BINARY_OP(/, div)
    //
    // minus
    ad_t operator -(void) const;
    //
    // sum
    ad_t sum(const c10::ArrayRef<long>& dim = c10::ArrayRef<long>() ) const;
    //
    // view
    ad_t view(const c10::ArrayRef<long>& shape) const;
    //
}; }
