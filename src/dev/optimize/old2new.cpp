// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin old2new dev}
{xrst_spell
    std
    adfn
    dev
}

Mapping From Old to New AD Function Tensor Indices
##################################################

Syntax
******
{xrst_code cpp}
auto [old2new_con, old2new_par, old2new_var] = dev::old2new(
   n_dom_par, n_dom_var, depend_con, depend_par, depend_var
)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_OLD2NEW, END_OLD2NEW
}

not_used
********
We use the notation not_used for the value
{xrst_code cpp}
    not_used = std::numeric_limits<size_t>::max()
{xrst_code}

n_dom_par
*********
The number of domain parameters in the old (and new) AD Function.

n_dom_var
*********
The number of domain variables in the old (and new) AD Function.

depend_con
**********
The length of this vector is the number of constants in the old AD function.
If depend_con[k] is true (false), the adfn :ref:`adfn_get_range@range`
depends (does not depend) in the k-th constant in the old AD function..

depend_par
**********
The length of this vector is the number of parameter in
:ref:`adfn_forward_par@par_all` for the old AD function.
If depend_par[k] is true (false), the range
depends (does not depend) in the k-th parameter in par_all.

depend_var
**********
The length of this vector is the number of variables in
:ref:`adfn_forward_var@var_all` for the old AD function.
If depend_var[k] is true (false), the range
depends (does not depend) in the k-th variable in var_all.

old2new_con
***********
The length of this vector is the number of constants in the old AD function.
If the k-th constant in the old function is used in the new function,
old2new_con[k] is its index in the new function
(the constants that are included have the same order).
If the k-th constant in the old function is not used in the new function,
old2new_con[k] == not_used.

old2new_par
***********
The length of this vector is the number of parameters in the old AD function.
If the k-th parameters in the old function is used in the new function,
old2new_par[k] is its index in the new function
(the parameters that are included have the same order).
If the k-th parameters in the old function is not used in the new function,
old2new_par[k] == not_used.

old2new_var
***********
The length of this vector is the number of variables in the old AD function.
If the k-th variables in the old function is used in the new function,
old2new_var[k] is its index in the new function
(the variables that are included have the same order).
If the k-th variables in the old function is not used in the new function,
old2new_var[k] == not_used.


{xrst_end old2new}
*/
#include <limits>
#include <array>
#include <ad_tensor/dev/optimize.hpp>
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
// BEGIN_OLD2NEW
std::array< vector<size_t>, 3 > old2new(
    size_t              n_dom_par,
    size_t              n_dom_var,
    const vector<bool>& depend_con,
    const vector<bool>& depend_par,
    const vector<bool>& depend_var)
{   // END_OLD2NEW
    //
    // not_used
    size_t not_used = std::numeric_limits<size_t>::max();
    //
    // old2new_con
    // The first constant is a empty_at_ten() (keep even if it is not used).
    vector<size_t> old2new_con(depend_con.size(), not_used);
    size_t n_con    = depend_con.size();
    old2new_con[0]  = 0;
    size_t i_new    = 1;
    for(size_t i_old = 1; i_old < n_con; ++i_old) {
        if(depend_con[i_old]) {
            old2new_con[i_old] = i_new;
            ++i_new;
        }
    }
    //
    // old2new_par
    vector<size_t> old2new_par(depend_par.size(), not_used);
    for(size_t j = 0; j < n_dom_par; ++j) {
        old2new_par[j] = j;
    }
    i_new           = n_dom_par;
    size_t n_par    = depend_par.size();
    for(size_t i_old = n_dom_par; i_old < n_par; ++i_old) {
        if(depend_par[i_old]) {
            old2new_par[i_old] = i_new;
            ++i_new;
        }
    }
    //
    // old2new_var
    vector<size_t> old2new_var(depend_var.size(), not_used);
    for(size_t j = 0; j < n_dom_var; ++j) {
        old2new_var[j] = j;
    }
    i_new           = n_dom_var;
    size_t n_var    = depend_var.size();
    for(size_t i_old = n_dom_var; i_old < n_var; ++i_old) {
        if(depend_var[i_old]) {
            old2new_var[i_old] = i_new;
            ++i_new;
        }
    }
    return std::array< vector<size_t>, 3 >(
       { old2new_con, old2new_par, old2new_var }
    );
}
//
} } // END_AD_TENSOR_DEV_NAMESPACE
