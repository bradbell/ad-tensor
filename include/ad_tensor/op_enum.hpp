// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin op_enum dev}

The Scoped Operator Enum Type
#############################

Enum Values
***********
{xrst_literal ,
    BEGIN_SORT, END_SORT
}
{xrst_end op_enum}
*/
// BEGIN_SORT_THIS_LINE_PLUS_3
namespace ad_tensor {
    enum class op_enum {
        add, // addition
        div, // division
        mul, // multiplication
        sub, // subtraction
    };
}
// END_SORT_THIS_LINE_MINUS_3
