#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin op_enum dev}

The Operator Scoped Enum Type
#############################

op_enum_t
*********
{xrst_literal ,
    BEGIN_OP_ENUM_T, END_OP_ENUM_T
}

Unary Operators
***************
{xrst_literal ,
    BEGIN_UNARY, END_UNARY
}

Binary Operators
****************
{xrst_literal ,
    BEGIN_BINARY, END_BINARY
}

Other Operators
***************
{xrst_literal ,
    BEGIN_OTHER, END_OTHER
}

{xrst_end op_enum}
*/
// BEGIN_OP_ENUM_T
namespace ad_tensor { namespace dev { enum struct op_enum_t
{   // END_OP_ENUM_T
    //
    // BEGIN_UNARY BEGIN_SORT_THIS_LINE_PLUS_1
    exp,         // exponentiation
    inverse,     // maxtrix inverse
    logdet,      // log of determinant
    // END_UNARY END_SORT_THIS_LINE_MINUS_1
    //
    // BEGIN_BINARY BEGIN_SORT_THIS_LINE_PLUS_1
    add,         // addition
    div,         // division
    eq,          // equal
    ge,          // greater than or equal
    gt,          // greater than
    le,          // less than or equal
    lt,          // less than
    mul,         // multiplication
    ne,          // not equal
    sub,         // subtraction
    // END_BINARY  END_SORT_THIS_LINE_MINUS_1
    //
    // BEGIN_OTHER  BEGIN_SORT_THIS_LINE_PLUS_1
    call,        // call atomic function
    call_result, // place holder when more than one result for a call
    dom,         // domain tensor
    index_put,   // assign elements of a tensor
    matmul,      // matrix multiplication
    minus,       // unary minus
    solve,       // solve matrix equation
    sum,         // summation
    transpose,   // transpose
    view,        // view
    where,       // where function
    // END_OTHER END_SORT_THIS_LINE_MINUS_1
}; } }
