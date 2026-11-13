#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adtype dev}
{xrst_spell
    enum
    adfn
}

A Graph Index Type
##################

constant
********
a constant index in the m_con_vec vector for the
tape that holds this graph.

parameter
*********
a parameter index in the :ref:`adfn_forward_par@par_all` vector
for the adfn that holds this graph.

variable
********
a variable index in the :ref:`adfn_forward_var@var_all` vector
for the adfn that holds this graph.

none
****
The type none is used when the index does not correspond to an AD tensor.

Enum Values
***********
{xrst_literal ,
    BEGIN_AD_TYPE_T, END_AD_TYPE_T
}

{xrst_end adtype}
*/
// BEGIN_AD_TYPE_T
namespace ad_tensor { enum struct adtype_t {
    constant,
    parameter,
    variable,
    none,      // This does not correspond to an AD tensor
}; }
// END_AD_TYPE_T
