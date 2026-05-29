// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/get_option.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { namespace dev { // BEGIN_NAMESPACE_AD_TENSOR_DEV
//
// get_option
std::string get_option(
    const options_t&                options         ,
    const std::string&              key             ,
    const std::string&              default_value   ,
    const std::set<std::string>&    other_values    )
{   //
    std::string value = default_value;
    for( auto itr = options.begin(); itr != options.end(); ++itr)
    {   if( itr->first == key ) {
            value = itr->second;
            if( value != default_value && ! other_values.empty() ) {
                if( other_values.find(value) == other_values.end() ) {
                    std::string msg = "get_option";
                    msg += ": key = " + key;
                    msg += ", value = " + value;
                    msg += ", possible = { " + default_value;
                    auto begin = other_values.begin();
                    auto end   = other_values.end();
                    for(auto tmp_itr = begin; tmp_itr != end; ++tmp_itr) {
                        msg += ", " + *tmp_itr;
                    }
                    msg += " }\n";
                    msg += "value is not in possible set for this key";
                    user_assert(false , msg);
                }
            }
        }
    }
    return value;
}
//
} } // END_NAMESPACE_AD_TENSOR_DEV
