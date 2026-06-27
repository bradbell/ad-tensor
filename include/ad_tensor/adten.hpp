#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
/*
------------------------------------------------------------------------------
{xrst_begin adten usr}
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

Constructor
***********
{xrst_literal ,
    BEGIN_PUBLIC_CTOR, END_PUBLIC_CTOR
}
The default constructor uses an empty tensor.
If the tensor used in the construction is non-empty,
and a tape is currently being recorded,
the AD tensor is linked to a new constant in the current tape.

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
where lhs and rhs are const adten_t reference

Compound Assignment Operators
*****************************
For *op* equal ``+=``, ``-=``, ``*=``, ``/=`` :
{xrst_code cpp}
    lhs op rhs
{xrst_code}
where lhs and rhs are adten_t references and rhs is.

Other Member Functions
**********************
{xrst_comment BEGIN_SORT_THIS_LINE_PLUS_2}
{xrst_toc_table
    src/adten/matmul.cpp
    src/adten/minus.cpp
    src/adten/recording.cpp
    src/adten/solve.cpp
    src/adten/sum.cpp
    src/adten/transpose.cpp
    src/adten/view.cpp
}
{xrst_comment END_SORT_THIS_LINE_MINUS_2}

Example
*******
{xrst_literal ,
    examples/adten/binary.cpp
    BEGIN_CPP, // END_CPP
}

{xrst_end adten}
------------------------------------------------------------------------------
{xrst_begin_parent adten_dev dev}

The adten_t Class Developer Documentation
#########################################
{xrst_comment BEGIN_SORT_THIS_LINE_PLUS_2}
{xrst_toc_table after
    src/adten/binary.cpp
    src/adten/matmul.cpp
    src/adten/minus.cpp
    src/adten/solve.cpp
    src/adten/sum.cpp
    src/adten/transpose.cpp
    src/adten/view.cpp
}
{xrst_comment END_SORT_THIS_LINE_MINUS_2}

{xrst_end adten_dev}
------------------------------------------------------------------------------
{xrst_begin adten_member dev}

The adten_t Member Data
#######################
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

{xrst_end adten_member}
------------------------------------------------------------------------------
{xrst_begin adten_ctor_dev dev}

The adten_t Private Constructor
###############################
{xrst_literal ,
    BEGIN_PRIVATE_CTOR, END_PRIVATE_CTOR
}

{xrst_end adten_ctor_dev}
------------------------------------------------------------------------------
*/
#include <torch/torch.h>
//
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/dev/op_enum.hpp>
//
// AD_TENSOR_BINARY_OP
# define AD_TENSOR_BINARY_OP(op, op_enum) \
    \
    adten_t operator op (const adten_t& rhs) const \
    { return binary( dev::op_enum_t:: op_enum, *this, rhs ); } \
    \
    adten_t& operator op ## = (const adten_t& rhs) { \
        *this = *this op rhs; \
        return *this; \
    }
//
// BEGIN_AD_CLASS
namespace ad_tensor { class adten_t
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
    adten_t(
        size_t tape_id,
        size_t index,
        const at::Tensor& tensor,
        ad_type_t ad_type )
    : m_tape_id(tape_id), m_index(index), m_tensor(tensor), m_ad_type(ad_type)
    { }
    // END_PRIVATE_CTOR
    //
    // binary
    static adten_t binary(
        dev::op_enum_t op_enum, const adten_t& lhs, const adten_t& rhs
    );
public:
    // BEGIN_PUBLIC_CTOR
    adten_t(void);
    explicit adten_t( const at::Tensor& tensor );
    // END_PUBLIC_CTOR
    //
    // BEGIN_SIZES
    c10::IntArrayRef sizes(void) const
    // END_SIZES
    {   return m_tensor.sizes(); }
    //
    // BEGIN_NUMEL
    int64_t numel(void) const
    // END_NUMEL
    {   return m_tensor.numel(); }
    //
    // BEGIN_TO_TENSOR
    const at::Tensor& tensor(void) const
    // END_TO_TENSOR
    {   return m_tensor; }
    //
    // BEGIN_START_RECORDING_1
    // adom_var = adten_t::start_recording(dom_var)
    static vector<adten_t> start_recording(
        const vector<at::Tensor>& dom_var
    );
    // END_START_RECORDING_1
    // BEGIN_START_RECORDING_2
    // auto [adom_var  adom_par] = adten_t::start_recording(dom_var, dom_par)
    static std::tuple< vector<adten_t>, vector<adten_t> > start_recording(
        const vector<at::Tensor>& dom_var ,
        const vector<at::Tensor>& dom_par
    );
    // END_START_RECORDING_2
    //
    static adfn_t stop_recording(
        const vector<adten_t>& arange ,
        const std::string&     name
    );
    //
    // Binary operators
    AD_TENSOR_BINARY_OP(+, add)
    AD_TENSOR_BINARY_OP(-, sub)
    AD_TENSOR_BINARY_OP(*, mul)
    AD_TENSOR_BINARY_OP(/, div)
    //
    // minus
    adten_t operator -(void) const;
    //
    // sum
    adten_t sum(const c10::IntArrayRef& dim = c10::IntArrayRef() ) const;
    //
    // view
    adten_t view(const c10::IntArrayRef& shape) const;
    //
    // matmul
    adten_t matmul(const adten_t& rhs) const;
    //
    // transpose
    adten_t transpose(int64_t dim1, int64_t dim2) const;
    //
    // solve
    adten_t solve(const adten_t& rhs, bool left) const;
    //
    // call_atom
    static vector<adten_t> call_atom(
        size_t                 atom_id   ,
        size_t                 call_info ,
        const vector<adten_t>& adomain
    );
}; }

namespace ad_tensor {
    //
    // linalg_solve
    adten_t linalg_solve(
        const adten_t& square, const adten_t& rhs, bool left = true
    );
}
