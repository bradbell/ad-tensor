// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
-------------------------------------------------------------------------------
{xrst_begin adten_solve usr}
{xrst_spell
    rhs
}

Solve Linear Equations
######################

Prototype
*********
{xrst_literal ,
    BEGIN_LINALG_SOLVE, END_LINALG_SOLVE
}

square
******
is a two dimensional invertible square matrix.

rhs
***
is the right hand side of the matrix equation.
This must be one or two dimensional.
If it is one dimensional, it is treated as a column vector.

left
****
if true (false) the square matrix is on the left (right).
The default for left is true.

solution
********
is the solution of the linear equation

    square   x solution = rhs      (left true)
    solution x square   = rhs      (left false)

where x denotes matrix multiplication.

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
    BEGIN_SOLVE, END_SOLVE
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
    start + 0, index for square,           type for square
    start + 1, index for rhs,              type for rhs
    start + 2, 1 (0) is left true (false), ad_type::none

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

// BEGIN_SOLVE
adten_t adten_t::solve(const adten_t& rhs, bool left) const
// END_SOLVE
{
    //
    // square
    const adten_t& square = *this;
    //
# ifndef NDEBUG
    size_t square_n_dim = this->sizes().size();
    size_t rhs_n_dim    = rhs.sizes().size();
    //
    dev::user_assert( square_n_dim == 2 ,
        "solve: square is not two dimensional"
    );
    dev::user_assert( rhs_n_dim == 1 || rhs_n_dim == 2 ,
        "solve: rhs is not one or two dimensional"
    );
    int64_t square_n_row = square.sizes()[0];
    int64_t square_n_col = square.sizes()[1];
    dev::user_assert( square_n_row != square_n_col ,
        "solve: square: number of rows is not equal number of columns"
    );
    int64_t rhs_n_row = rhs.sizes()[0];
    int64_t rhs_n_col = 1;
    if( rhs_n_dim == 2 ) {
        rhs_n_col = rhs.sizes()[1];
    }
    dev::user_assert( square_n_row != rhs_n_row || ! left , "solve: "
        "left true and number of rows in square and rhs are not equal"
    );
    dev::user_assert( square_n_col != rhs_n_col || left , "solve: "
        "left false and number of columns in square and rhs are not equal"
    );
# endif
    // res_tensor
    at::Tensor res_tensor = at::linalg_solve(m_tensor, rhs.m_tensor, left);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id , "solve: "
        "square AD tensor does not match tape that is recording"
    );
    dev::user_assert( rhs.m_tape_id == tape.m_tape_id , "solve: "
        "rhs AD tensor does not match tape that is recording"
    );
    //
    // res_ad_type
    ad_type_t res_ad_type = std::max( m_ad_type, rhs.m_ad_type );
    //
    // res_tape_id
    size_t res_tape_id = tape.m_tape_id;
    //
    // res_index
    size_t res_index;
    //
    if( res_ad_type == ad_type_t::constant ) {
        // res_index, tape.m_con
        res_index = tape.m_con.size();
        tape.m_con.push_back( res_tensor.clone() );
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_ad_type == ad_type_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_ad_type == ad_type_t::variable  && "AD tensor in "
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
        agraph->m_arg_type.push_back( m_ad_type );
        //
        agraph->m_arg_value.push_back( rhs.m_index );
        agraph->m_arg_type.push_back( rhs.m_ad_type );
        //
        agraph->m_arg_value.push_back( size_t(left) );
        agraph->m_arg_type.push_back( ad_type_t::none );
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_ad_type);
}
// BEGIN_LINALG_SOLVE
// solution = linalg_solve(square, rsh, left)
adten_t linalg_solve(const adten_t& square, const adten_t& rhs, bool left)
// END_LINALG_SOLVE
{   return square.solve(rhs, left);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
