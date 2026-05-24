#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin op_enum dev}

The Operator Scoped Enum Type
#############################

Enum Values
***********
{xrst_literal ,
    BEGIN_SORT, END_SORT
}
{xrst_end op_enum}
*/
// BEGIN_SORT_THIS_LINE_PLUS_2
namespace ad_tensor { namespace dev { enum struct op_enum_t {
    add,  // addition
    div,  // division
    dom,  // domain tensor
    mul,  // multiplication
    sub,  // subtraction
    test, // used for testing
}; } }
// END_SORT_THIS_LINE_MINUS_2
