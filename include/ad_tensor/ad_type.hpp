#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin ad_type usr}
{xrst_spell
    enum
}

The Types Corresponding to AD tensors
#####################################

constant
********
a constant tensor does not depend on the
domain parameter tensor :ref:`start_recording@dom_par` or
the domain variable tensor :ref:`start_recording@dom_var` .
The elements of a constant tensor are called constants.

parameter
*********
a parameter tensor depends on the
domain parameter tensor but not on the domain variable tensor.
The elements of a parameter tensor are called parameters.

variable
********
a variable tensor depends on the domain variable tensor.
The elements of a variable tensor are called variables.

none
****
The type none is used when the ad_type does not correspond to an AD tensor.

Enum Values
***********
{xrst_literal ,
    BEGIN_AD_TYPE_T, END_AD_TYPE_T
}

{xrst_end ad_type}
*/
// BEGIN_AD_TYPE_T
namespace ad_tensor { enum struct ad_type_t {
    constant,
    parameter,
    variable,
    none,      // This does not correspond to an AD tensor
}; }
// END_AD_TYPE_T
