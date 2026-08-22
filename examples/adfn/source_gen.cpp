// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_source_gen usr}
{xrst_spell
    std
    dll
}

Generate C++ Source Code For this AD Function
#############################################

Under Construction
******************
The documentation below is just a plan.
Its implementation is under construction.

Syntax
******
{xrst_code cpp}
    source_code = adfn.source_gen()
{xrst_code}

adfn
****
We use adfn for this AD function object.

source_code
***********
Is C++ source code for the function range = adfn(dom_var, dom_par) where:
{xrst_code cpp}
    par_all = adfn.forward_par(dom_par);
    var_all = adfn.forward_var(dom_var, par_all);
    y       = adfn.get_range(var_all, par_all);
{xrst_code}
The vector y above can contain derivatives if adfn
was recording using AD Tensors and derivatives of another AD function.
The C interface to the source code function is defined below.
(A C interface is used to make dll linking simpler.)

C Interface
***********

Syntax
======
{xrst_code cpp}
   auto [ range, message ] = adfn_name_plugin(dom_par, dom_var )
{xrst_code}

adfn_name_plugin
----------------
This begins with the name of this adfn; see :ref:`stop_recording-name`
and the ends with ``_plugin`` .

dom_par
-------
is the vector of domain parameter tensors and has the following prototype:
{xrst_code cpp}
    const ad_tensor::vector<at::Tensor>& dom_par;
{xrst_code}

dom_var
-------
is the vector of domain variable tensors and has the following prototype:
{xrst_code cpp}
    const ad_tensor::vector<at::Tensor>& dom_var;
{xrst_code}

range
-----
is the vector of range tensors and has the following prototype:
{xrst_code cpp}
    ad::tensor::vector<at::Tensor> range
{xrst_code}
If a fatal error occurs, range will be the empty vector.

message
-------
is the warning or error message and has the following prototype:
{xrst_code cpp}
    std::string message;
{xrst_code}
If message is empty, no error or warnings occurred.

{xrst_end adfn_source_gen}
*/
