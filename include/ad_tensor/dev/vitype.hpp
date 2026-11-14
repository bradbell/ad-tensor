#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin vitype dev}
{xrst_spell
    enum
}

The Vector Index Type
#####################

Enum Values
***********

constant
========
an index in the m_con_vec vector for the
tape that holds this graph.

parameter
=========
corresponding value is an index in the :ref:`adfn_forward_par@par_all` vector
for the tape that holds this graph.

variable
========
corresponding value is an index in the :ref:`adfn_forward_var@var_all` vector
for the tape that holds this graph.

int64vector
===========
corresponding value is an index in the m_int64_vec vector
for the tape that holds this graph.

none
====
The type none is used this is not an index in a
vector for the tape that holds this graph.


AD Tensor Types
***************
The AD Tensor types are constant, parameter, and variable; see
:ref:`start_recording@AD Tensor Types@Constant`,
:ref:`start_recording@AD Tensor Types@Parameter`, and
:ref:`start_recording@AD Tensor Types@Variable` .
They satisfy the following relation:

    constant < parameter < variable

{xrst_end vitype}
*/
// BEGIN_AD_TYPE_T
namespace ad_tensor { enum struct vitype_t {
    constant,
    parameter,
    variable,
    int64vector,
    none,
}; }
// END_AD_TYPE_T
