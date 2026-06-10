#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin get_option dev}

Get Option Value for Key
########################

Prototype
*********
{xrst_literal ,
    BEGIN_GET_OPTION, END_GET_OPTION
}

options
*******
contains the choices for all the keys.

key
***
is the key we are getting the choice for.

default_value
*************
is the default value to use if the key is not in options.

other_values
************
if this set is empty, there are no restrictions in the value for this key.
Otherwise, the value for this key must be the default or in this set.

value
*****
The return is the option choice for this key.

{xrst_end get_option}
*/
#include <set>
#include <ad_tensor/options.hpp>
//
namespace ad_tensor { namespace dev {
    //
    // BEGIN_GET_OPTION
    // value = ad_tensor::get_option(options, key, default_value)
    // value = ad_tensor::get_option(options, key, default_value, other_values)
    std::string get_option(
        const options_t&  options        ,
        const std::string&           key            ,
        const std::string&           default_value  ,
        const std::set<std::string>& other_values   = std::set<std::string>()
    );
    // END_GET_OPTION
} }
