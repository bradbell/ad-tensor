#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
/*
------------------------------------------------------------------------------
{xrst_begin_parent adten usr}
{xrst_spell
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

at_ten
******
This returns the at:Tensor corresponding to this object.
{xrst_literal ,
    BEGIN_AT_TEN, END_AT_TEN
}

sizes, numel
************
For *fun* equals ``sizes``, ``numel``,
return the result for the underlying at::Tensor.
{xrst_literal ,
    BEGIN_SIZES, END_SIZES
    BEGIN_NUMEL, END_NUMEL
}

clone
*****
Create a new adten_t where the underlying at::Tensor
has also been cloned.
{xrst_literal ,
    BEGIN_CLONE, END_CLONE
}

Other Member Functions
**********************
{xrst_comment BEGIN_SORT_THIS_LINE_PLUS_2}
{xrst_toc_table after
    src/adten/index_put.cpp
    src/adten/matmul.cpp
    src/adten/minus.cpp
    src/adten/recording.cpp
    src/adten/solve.cpp
    src/adten/sum.cpp
    src/adten/transpose.cpp
    src/adten/view.cpp
    src/adten/where.cpp
}
{xrst_comment END_SORT_THIS_LINE_MINUS_2}

{xrst_end adten}
------------------------------------------------------------------------------
{xrst_begin adten_unary usr}
{xrst_spell
    exp
}


AD Tensor Unary Operators
#########################

Syntax
******
{xrst_code cpp}
    result = adten.unary_function()
{xrst_code}
where adten is a ``const aten_t&`` and unary_function is one of the following:
``exp`` .

Example
*******
{xrst_comment BEGIN_SORT_THIS_LINE_PLUS_2}
{xrst_toc_table
    examples/adten/unary/exp.cpp
    examples/adten/unary/inverse.cpp
    examples/adten/unary/logdet.cpp
}
{xrst_comment END_SORT_THIS_LINE_MINUS_2}

{xrst_end adten_unary}
------------------------------------------------------------------------------
{xrst_begin adten_binary usr}
{xrst_spell
    lhs
    rhs
    op
}

AD Tensor Binary Operators
##########################

Syntax
******
{xrst_code cpp}
    result = lhs op rhs
{xrst_code}
where lhs and rhs are ``const adten_t&`` ,
result is ``adten_t`` and op is one of the following:

.. csv-table::
    :header-rows: 1

    op , description
   \+  ,  addition
   \-  ,  subtraction
   \*  ,  multiplication
    /  ,  division
    <  ,  less than
    <= , less than or equal
    == , equal
    != , not equal
    >= , greater than or equal
    >  ,  greater than

Example
*******
{xrst_literal ,
    examples/adten/binary.cpp
    BEGIN_CPP, END_CPP
}

{xrst_end adten_binary}
------------------------------------------------------------------------------
{xrst_begin adten_compound usr}
{xrst_spell
    lhs
    rhs
    op
}


AD Tensor Compound Assignment Operators
#######################################

Syntax
******
{xrst_code cpp}
    lhs op rhs
{xrst_code}
where lhs is an ``adten_t&`` , rhs is a ``const adten_t&`` , and
op is one of the following: ``+=``, ``-=``, ``*=``, ``/=`` .


Example
*******
{xrst_literal ,
    examples/adten/compound.cpp
    BEGIN_CPP, END_CPP
}

{xrst_end adten_compound}
------------------------------------------------------------------------------
{xrst_begin_parent adten_dev dev}

The adten_t Class Developer Documentation
#########################################
{xrst_comment BEGIN_SORT_THIS_LINE_PLUS_2}
{xrst_toc_table after
    src/adten/binary.cpp
    src/adten/index_put.cpp
    src/adten/matmul.cpp
    src/adten/minus.cpp
    src/adten/solve.cpp
    src/adten/sum.cpp
    src/adten/transpose.cpp
    src/adten/unary.cpp
    src/adten/view.cpp
    src/adten/where.cpp
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

m_adtype
********
If this thread matches the current recording,
m_adtype is the :ref:`adtype-name` in this threads tape for this AD tensor.

m_at_ten
********
is the at::Tensor corresponding to this AD tensor.

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
#include <tuple>
#include <torch/torch.h>
//
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/adtype.hpp>
#include <ad_tensor/dev/op_enum.hpp>
//
// AD_TENSOR_UNARY_OP
# define AD_TENSOR_UNARY_OP(unary_function) \
    adten_t unary_function(void) const \
    {   return unary( dev::op_enum_t:: unary_function, *this); }
//
// AD_TENSOR_BINARY_OP
# define AD_TENSOR_BINARY_OP(op, op_enum) \
    adten_t operator op (const adten_t& rhs) const \
    { return binary( dev::op_enum_t:: op_enum, *this, rhs ); }
//
// AD_TENSOR_COMPOUND_ASSIGNMENT
# define AD_TENSOR_COMPOUND_ASSIGNMENT(op) \
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
    // ad_tensor::call_atom
    // is a link to the private function call_atom defined in this class
    friend vector<adten_t> call_atom(
        size_t                 atom_id   ,
        const vector<adten_t>& adomain
    );
    // BEGIN_MEMBER_DATA
    size_t        m_tape_id;
    size_t        m_index;
    at::Tensor    m_at_ten;
    adtype_t     m_adtype;
    // END_MEMBER_DATA
    //
    // BEGIN_PRIVATE_CTOR
    adten_t(
        size_t tape_id,
        size_t index,
        const at::Tensor& tensor,
        adtype_t adtype )
    : m_tape_id(tape_id), m_index(index), m_at_ten(tensor), m_adtype(adtype)
    { }
    // END_PRIVATE_CTOR
    //
    // unary
    static adten_t unary(
        dev::op_enum_t op_enum, const adten_t& operand
    );
    //
    // binary
    static adten_t binary(
        dev::op_enum_t op_enum, const adten_t& lhs, const adten_t& rhs
    );
    //
    // call_atom
    static vector<adten_t> call_atom(
        size_t                 atom_id   ,
        const vector<adten_t>& adomain
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
    {   return m_at_ten.sizes(); }
    //
    // BEGIN_NUMEL
    int64_t numel(void) const
    // END_NUMEL
    {   return m_at_ten.numel(); }
    //
    // BEGIN_CLONE
    adten_t clone(void) const
    // END_CLONE
    {   return adten_t(m_tape_id, m_index,  m_at_ten.clone(), m_adtype ); }
    //
    // BEGIN_AT_TEN
    const at::Tensor& at_ten(void) const
    // END_AT_TEN
    {   return m_at_ten; }
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
    // BEGIN_WHERE
    static adten_t where(
        const adten_t& cond      ,
        const adten_t& true_case ,
        const adten_t& false_case
    );
    // END_WHERE
    //
    // Compound Assignment operators
    AD_TENSOR_COMPOUND_ASSIGNMENT(+)
    AD_TENSOR_COMPOUND_ASSIGNMENT(-)
    AD_TENSOR_COMPOUND_ASSIGNMENT(*)
    AD_TENSOR_COMPOUND_ASSIGNMENT(/)
    //
    // Numeric unary operators
    AD_TENSOR_UNARY_OP(exp);
    AD_TENSOR_UNARY_OP(logdet);
    AD_TENSOR_UNARY_OP(inverse);
    //
    // Numeric binary operators
    AD_TENSOR_BINARY_OP(+, add)
    AD_TENSOR_BINARY_OP(-, sub)
    AD_TENSOR_BINARY_OP(*, mul)
    AD_TENSOR_BINARY_OP(/, div)
    //
    // Compare binary operators
    AD_TENSOR_BINARY_OP(<,  lt)
    AD_TENSOR_BINARY_OP(<=, le)
    AD_TENSOR_BINARY_OP(==, eq)
    AD_TENSOR_BINARY_OP(!=, ne)
    AD_TENSOR_BINARY_OP(>=, ge)
    AD_TENSOR_BINARY_OP(>,  gt)
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
    // index_put
    adten_t index_put(
        const c10::List< std::optional<at::Tensor> >& input ,
        const adten_t&                                replace
    ) const;
}; }

namespace ad_tensor {
    //
    // linalg_solve
    adten_t linalg_solve(
        const adten_t& linear, const adten_t& rhs, bool left = true
    );
}
