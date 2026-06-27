// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
# include <ad_tensor/options.hpp>
//
/*
{xrst_begin options usr}

The Options Key Value Map
#########################

Constructor
***********
{xrst_literal ,
    BEGIN_CTOR, END_CTOR
}

name
****
{xrst_literal ,
    BEGIN_SET_NAME, END_SET_NAME
    BEGIN_GET_NAME, END_GET_NAME
}
The default value for name is the empty string.

trace
*****
{xrst_literal ,
    BEGIN_SET_TRACE, END_SET_TRACE
    BEGIN_GET_TRACE, END_GET_TRACE
}
The default value for trace is false.


{xrst_end options}
*/
namespace ad_tensor {
    //
    // BEGIN_CTOR
    // options_t options;
    options_t::options_t(void)
    // END_CTOR
    : m_name("")
    , m_trace(false)
    { }
    //
    // BEGIN_SET_NAME
    // options.set_name(name)
    void options_t::set_name(const std::string& name)
    {   // END_SET_NAME
        m_name = name;
    }
    // BEGIN_GET_NAME
    // name = options.get_name()
    const std::string& options_t::get_name(void) const
    {   // END_GET_NAME
        return m_name;
    }
    //
    // BEGIN_SET_TRACE
    // options.set_trace(trace)
    void options_t::set_trace(bool trace)
    {   // END_SET_TRACE
        m_trace = trace;
    }
    // BEGIN_GET_TRACE
    // trace = options.get_trace()
    bool options_t::get_trace(void) const
    {   // END_GET_TRACE
        return m_trace;
    }
}
