// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
-------------------------------------------------------------------------------
{xrst_begin adten_solve usr}
{xrst_spell
    rhs
    linalg
}

Solve Linear Equations
######################

Syntax
******
{xrst_code cpp}
    unknown = linalg_solve(linear, rhs, flag)
{xrst_code}

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_LINALG_SOLVE, END_LINALG_SOLVE
}

linear
******
is a square matrix or (\*,n,n) batch of square matrices,
that define the linear equations.
If it is a batch of matrices, broadcasting is used to match
the rhs dimensions.

rhs
***
is a square matrix or (\*,n,n) batch of square matrices,
that define the right hand sides in the equations.
If it is a batch of matrices, broadcasting is used to match
the linear dimensions.

left
****
if true (false) the linear matrix is on the left (right).

unknown
*******
is the matrix unknown (or batch of matrices) containing the
unknowns in the linear equation

    linear   x unknown  = rhs      (left true)
    unknown  x linear   = rhs      (left false)

where x denotes matrix multiplication.

Example
*******
{xrst_toc_table 
    examples/adten/solve.cpp
}
{xrst_end adten_solve}
-------------------------------------------------------------------------------
{xrst_begin adten_solve_dev dev}
{xrst_spell
    rhs
}

Compute and Record Solution of Linear Equations
###############################################

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_LINALG_SOLVE, END_LINALG_SOLVE
}

Recording
*********
If this thread's tape is recording, and the result is (is not) a constant,
the constant is added to the tape (the operation is added to the tape).

Operation
*********
If this thread's tape is recording and the result is a parameter (variable)
the following is added to the parameter (variable) acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value,                  arg_type
    start + 0, index for linear,           type for linear
    start + 1, index for rhs,              type for rhs
    start + 2, 1 (0) is left true (false), adtype::none

where start be the length of arg_value and arg_type before this call to
``adten_t::binary`` .

{xrst_end adten_solve_dev}
-------------------------------------------------------------------------------
*/
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/user_assert.hpp>

namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR

adten_t adten_t::solve(const adten_t& rhs, bool left) const
{
    //
# ifndef NDEBUG
    const adten_t&   linear = *this;
    c10::IntArrayRef linear_shape = linear.sizes();
    dev::user_assert( linear_shape.size() >= 2 ,
        "linalg_solve: linear is not two or more dimensional"
    );
    size_t n_linear = linear_shape.size();
    dev::user_assert( linear_shape[n_linear-1] == linear_shape[n_linear-2],
        "linalg_solve: linear is square matrix or batch of square matrices"
    );
    c10::IntArrayRef rhs_shape = rhs.sizes();
    dev::user_assert( rhs_shape.size() >= 2 ,
        "linalg_solve: rhs is not two or more dimensional"
    );
    size_t n_rhs    = rhs_shape.size();
    dev::user_assert( rhs_shape[n_rhs-1] == rhs_shape[n_rhs-2],
        "linalg_solve: rhs is square matrix or batch of square matrices"
    );
# endif
    // res_tensor
    at::Tensor res_tensor = at::linalg_solve(m_at_ten, rhs.m_at_ten, left);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id , "solve: "
        "linear AD tensor's tape is not tape that is recording"
    );
    dev::user_assert( rhs.m_tape_id == tape.m_tape_id , "solve: "
        "rhs AD tensor's tape is not tape that is recording"
    );
    //
    // res_adtype
    adtype_t res_adtype = std::max( m_adtype, rhs.m_adtype );
    //
    // res_tape_id
    size_t res_tape_id = tape.m_tape_id;
    //
    // res_index
    size_t res_index;
    //
    if( res_adtype == adtype_t::constant ) {
        // res_index, tape.m_con
        res_index = tape.m_con.size();
        tape.m_con.push_back( res_tensor.clone() );
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_adtype == adtype_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_adtype == adtype_t::variable  && "AD tensor in "
                "solve is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::solve );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( m_index );
        agraph->m_arg_type.push_back( m_adtype );
        //
        agraph->m_arg_value.push_back( rhs.m_index );
        agraph->m_arg_type.push_back( rhs.m_adtype );
        //
        agraph->m_arg_value.push_back( size_t(left) );
        agraph->m_arg_type.push_back( adtype_t::none );
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_adtype);
}
adten_t linalg_solve(const adten_t& linear, const adten_t& rhs, bool left)
{   return linear.solve(rhs, left);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
