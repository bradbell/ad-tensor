#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin op_base dev}

The Operator Base Class
#######################

Prototype
*********
{xrst_literal ,
    BEGIN_OP_BASE, END_OP_BASE
}

Derived Classes
***************
All the member functions for this class are public and virtual.
Derived classes are not intended to have member variables.
Their member functions are like static functions
that are grouped together by operator.

op_enum
*******
This function returns the :ref:`op_enum-name` value
for the corresponding operator.
{xrst_literal ,
    BEGIN_OP_ENUM, END_OP_ENUM
}

n_arg
*****
This function returns the number of arguments for this operator.
{xrst_literal ,
    BEGIN_N_ARG, END_N_ARG
}

offset
======
    is the index in *arg_all* and *ad_type_all* of
    the first argument for this use of this operator.

arg_all
=======
    is a vector containing the arguments for all the operator uses.

ad_type_all
===========
    is a vector containing the AD type for all the operator uses.

{xrst_end op_base}
*/
#include <vector>
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/ad_type.hpp>

// BEGIN_OP_BASE
namespace ad_tensor { namespace devel { class op_base_t
// END_OP_BASE
{
public:
    // op_enum
    // BEGIN_OP_ENUM
    virtual op_enum_t op_enum(void) const = 0;
    // END_OP_ENUM
    //
    // n_arg
    // BEGIN_N_ARG
    virtual size_t n_arg(
        size_t                        offset      ,
        const std::vector<size_t>&    arg_all     ,
        const std::vector<ad_type_t>& ad_type_all ) const = 0;
    // END_N_ARG

};
} }
