#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin base_op dev}
{xrst_spell
    der
    src
    newline
}

The Operator Base Class
#######################

Prototype
*********
{xrst_literal ,
    BEGIN_BASE_OP, END_BASE_OP
}

Derived Classes and Static Functions
************************************
All the member functions for this base class,
and its derived classes, are static or like static functions
that are grouped by derived class.
{xrst_toc_table
    include/ad_tensor/dev/derive_op.hpp
    src/dev/derive_op/src_gen_binary.cpp
    src/dev/derive_op/src_gen_unary.cpp
}

Common Arguments
****************
For each argument below, its documentation is common for
all the member functions where the corresponding argument appears:

op_index
========
is the index of this operator in the operation sequence.
It is also the index of the result for this operator.
It must be greater than zero, because the first result is an
independent parameter tensor.

agraph
======
Is the :ref:`agraph-name` for the operation sequence that we are using.

con_vec
=======
is the vector containing all the constant tensors.

par_vec
=======
is the vector containing all the parameter tensors.

var_vec
=======
is the vector containing all the variable tensors..

Pure Virtual Functions
**********************

forward_par
===========
For this function, arg_type[arg_index] is never variable.
In addition, par_vec[index] for index < op_index is an input
and par_vec[op_index] is an output.
{xrst_literal ,
    BEGIN_FORWARD_PAR, END_FORWARD_PAR
}

forward_var
===========
For this function,
var_vec[index] for index < op_index is an input
and var_vec[op_index] is an output.
{xrst_literal ,
    BEGIN_FORWARD_VAR, END_FORWARD_VAR
}

forward_der
===========
On input, for_der contains the derivative w.r.t. the domain direction
of all the variables up to but no including op_index; hence,
for_der[index] for index < op_index is an input to this routine.
Upon return for_der[op_index] contains the derivative w.r.t
the domain direction of the variable corresponding to op_index.
{xrst_literal ,
    BEGIN_FORWARD_DER, END_FORWARD_DER
}

reverse_der
===========
{xrst_literal ,
    BEGIN_REVERSE_DER, END_REVERSE_DER
}
#.  On input, for_der contains the derivative of the range direction summation
    w.r.t the variables from index zero to index op_index.
    Hence for_der[index] for index <= op_index are inputs to this routine.
#.  Upon return, for_der contains the derivative of the range direction
    summation w.r.t the variables from index zero to index op_index - 1.
    Hence for_der[index] for index < op_index are outputs to this routine.
    Actually all the outputs correspond to index values that are
    arguments to the operator at index op_index.
#.  The no element tensor corresponds to zero
    derivative for the corresponding variable. There is no reason to process
    cases where for_der[op_index] has no elements; hence for_der[op_index]
    has elements when this routine is called.

src_gen
=======
{xrst_literal ,
    BEGIN_SRC_GEN, END_SRC_GEN
}
The string returned by src_gen sets the value for the corresponding
operator, operator index, and acyclic graph.
It does not start or end with an newline and if it creates any new objects,
they must be inside a local scope; i.e.,  inside of {}.

variable_agraph
---------------
if this is true (false) the acyclic graph is for variables (parameters).

tensor_src
----------
The function call
{xrst_code cpp}
    src = tensor_src(index, adtype)
{xrst_code}
returns a source code representation of a
constant, parameter of variable (depending on adtype) with the specified index.
The return, src, can be used to set or get the corresponding value.
There are no newline characters in src.

{xrst_end base_op}
*/
#include <cassert>
#include <ad_tensor/vector.hpp>
#include <torch/torch.h>
//
#include <ad_tensor/adtype.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
//
// BEGIN_BASE_OP
namespace ad_tensor { namespace dev {
template<class TensorType> struct base_op_t
// END_BASE_OP
{   //
    // BEGIN_FORWARD_PAR
    virtual void forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<TensorType>&          par_vec
    ) const = 0;
    // END_FORWARD_PAR
    //
    // BEGIN_FORWARD_VAR
    virtual void forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        vector<TensorType>&          var_vec
    ) const = 0;
    // END_FORWARD_VAR
    //
    // BEGIN_FORWARD_DER
    virtual void forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          for_der
    ) const = 0;
    // END_FORWARD_DER
    //
    // BEGIN_REVERSE_DER
    virtual void reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          rev_der
    ) const = 0;
    // END_REVERSE_DER
    //
    // BEGIN_SRC_GEN
    virtual std::string src_gen(
        size_t                       op_index        ,
        const agraph_t&              agraph          ,
        bool                         variable_agraph ,
        const std::function< std::string(size_t, adtype_t) >& tensor_src
    ) const = 0;
    // END_SRC_GEN
};
} }
